from constants import *
import os
# import boto3
import json
import dataHandler
import subprocess
import shutil
#import PyPDF2
from PIL import Image, ImageDraw
import pdf2image
import cv2
# Removed SageMaker imports - now using local API
# from sagemaker.pytorch import PyTorchPredictor
# from sagemaker.deserializers import JSONDeserializer
import traceback
import requests
import time
import Levenshtein
from fpdf import FPDF
print("Finished imports")
# Initialize S3 client
# s3 = boto3.client('s3')

# directory corresponding to /mathsearch-cluster/lambda-container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def levenshtein_distance(query_string, latex_list, top_n):
  # elem of latex_list is (latex string, page num, eqn num)
  ranked_list = []
  n = len(latex_list)
  for i in range(n):
    latex1 = latex_list[i][0] # string is first element 
  
    similarity_score = Levenshtein.distance(latex1, query_string)
    ranked_list.append((latex_list[i][0], latex_list[i][1], latex_list[i][2], similarity_score))
  
  # Sort based on similarity score
  ranked_list.sort(key=lambda x: x[3])
  return ranked_list[:top_n]

# Returns a well-formatted LaTeX string represent the equation image 'image'
# Makes the MathPix API call
def image_to_latex_convert(image, query_bool):

    # Hardcoded CDS account response headers (placeholders for now)
    headers = {
        "app_id": "mathsearch_ff86f3_059645",
        "app_key": os.environ.get("APP_KEY")
    }
      
    # Declare api request payload (refer to https://docs.mathpix.com/?python#introduction for description)
    data = {
        "formats": ["latex_styled"], 
        "rm_fonts": True, 
        "rm_spaces": False,
        "idiomatic_braces": True
    }

    #print(f"type of img sent to mathpix {type(image)}, {query_bool}")
    #print(f"type after buffered reader stuff {type(io.BufferedReader(io.BytesIO(image)))}")
    # assume that image is stored in bytes
    response = requests.post("https://api.mathpix.com/v3/text",
                                files={"file": image},
                                data={"options_json": json.dumps(data)},
                                headers=headers)
       
    # Check if the request was successful
    if response.status_code == 200:
        #print("Successful API call!!")
        response_data = response.json()
        #print(json.dumps(response_data, indent=4, sort_keys=True))  # Print formatted JSON response
        return response_data.get("latex_styled", "")  # Get the LaTeX representation from the response, safely access the key
    else:
        print("Failed to get LaTeX on API call. Status code:", response.status_code)
        return ""

#print("Finished image_to_latex_convert")

# NEW METHOD! swapped out s3 buckets for local directories
# note that for a given uuid, "/mathsearch/input/{uuid}_pdf.pdf" and "/mathsearch/input/{uuid}_image.png" must exist
def download_files(pdf_name, query_name, png_converted_pdf_path, pdfs_from_bucket_path):
    local_pdf = os.path.join(pdfs_from_bucket_path, pdf_name + ".pdf")
    local_target_dir = f"{png_converted_pdf_path}_{pdf_name}"
    local_target = os.path.join(local_target_dir, "query.png")

    print("local_pdf",local_pdf)
    print("pdf_name",pdf_name)
    print("local_target", local_target)

    # ensure that the dirs exist
    os.makedirs(pdfs_from_bucket_path, exist_ok=True)
    os.makedirs(local_target_dir, exist_ok=True)

    # copy pdf from /math/search/input to /tmp
    src_pdf = os.path.join(INPUT_DIR, pdf_name + ".pdf")
    if not os.path.exists(src_pdf):
        raise FileNotFoundError(f"Expected input PDF not found: {src_pdf}")
    shutil.copy2(src_pdf, local_pdf)

    # convert pdf to png in /tmp
    images = pdf2image.convert_from_path(local_pdf, dpi=500)
    subprocess.run(f'mkdir -p {png_converted_pdf_path}_{pdf_name}', shell=True)
    for i,img in enumerate(images):
        pdf_image = f"{png_converted_pdf_path}_{pdf_name}/{i}.png"
        img.save(pdf_image)
    
    src_query = os.path.join(INPUT_DIR, query_name + ".png")
    if not os.path.exists(src_query):
        raise FileNotFoundError(f"Expected query image not found: {src_query}")
    shutil.copy2(src_query, local_target)

    return local_pdf, f"{png_converted_pdf_path}_{pdf_name}", local_target

#print("Finished download_files")

# Call draw_bounding_box on each PNG page of PDF
def draw_bounding_box(image_path_in, bounding_boxes):
  """"
  image_path_in : path to PNG which represents page from pdf
  bounding_boxes: list of list of bounding boxes
  """
  model_width, model_height = 640,640
  image = Image.open(image_path_in).convert('RGB')
  draw = ImageDraw.Draw(image)
  width, height = image.size
  x_ratio, y_ratio = width/model_width, height/model_height
  #SKYBLUE = (55,161,253)
  GREEN = (32,191,95)
  YELLOW = (255,225,101)

  # create rectangle for each bounding box on this page
  for bb, rank in bounding_boxes:
    x1, y1, x2, y2 = bb
    x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
    y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
    if rank == 0: 
      draw.rectangle(xy=(x1, y1, x2, y2), outline=GREEN, width=8)
    else:
      draw.rectangle(xy=(x1, y1, x2, y2), outline=YELLOW, width=8)
  
  return image
  # save img as pdf
  #image.save(image_path_out[:-4]+".pdf")

#print("Finished draw_bounding_box")

def final_output(pdf_name, png_pdf_path, bounding_boxes):
  """
  bounding_boxes : dict with keys page numbers, and values (list of bounding boxes, eqn rank)
  """
  IMG_IN_DIR = f"/tmp/converted_pdfs_{pdf_name}/"
  IMG_OUT_DIR = "/tmp/img_out/"
  subprocess.run(["rm", "-rf", IMG_OUT_DIR])
  subprocess.run(["mkdir", "-p", IMG_OUT_DIR])
  
  PDF_IN_DIR = "/tmp/pdfs_from_bucket/"
  PDF_OUT_DIR = "/tmp/pdf_out/"
  subprocess.run(["rm", "-rf", PDF_OUT_DIR])
  subprocess.run(["mkdir", "-p", PDF_OUT_DIR])

  pdf_in = PDF_IN_DIR + pdf_name + ".pdf"
  #pdf_out = PDF_OUT_DIR + pdf_name
  pdf_out_tmp = PDF_OUT_DIR + pdf_name[:-4]+".pdf"
  #pdf_no_ext = pdf_name[:-4]

  result_pages = list(bounding_boxes.keys())
  #print("bounding boxes dict: ", bounding_boxes)

  # call "draw_bounding_boxes" for each png page, save to IMG_OUT_DIR
  # merge the rendered images (with bounding boxes) to the pdf, and upload to S3
  paths = sorted(os.listdir(png_pdf_path))
  pdf = FPDF()
  for i in range(len(paths)-1):
    #print(f"adding {paths[i]}")
    if str(i) in result_pages:  
      full_path = os.path.join(png_pdf_path, paths[i])
      img = draw_bounding_box(full_path, ...)
      img.save(full_path)
      pdf.image(full_path)
    
    pdf.add_page()
    pdf.image(paths[i], 0, 0, 210, 297) # A4 paper sizing
  pdf.output(pdf_out_tmp, "F")

  ## TODO this will have to be changed to local solution
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  final_pdf_name = pdf_name[:-4] + ".pdf"
  final_pdf_path = os.path.join(OUTPUT_DIR, final_pdf_name)
  shutil.copy2(pdf_out_tmp, final_pdf_path)
  print(f"merged final pdf, saved to {final_pdf_path}")

  return final_pdf_path

#print("Finished final_output")

# Store string repr. of LaTeX equation and its page number in list
def rank_eqn_similarity(yolo_result, query_path, pdf_name):
  with open(query_path, "rb") as f:
    data = f.read()
    query_text = image_to_latex_convert(data, query_bool=True)
  query_text = query_text.replace(" ", "")
  print(f"query_text: {query_text}")
      
  equations_list = []
  for dict_elem, page_num in yolo_result:
    eqn_num = 1
    
    total_eqns = 0
    skipped_eqns = 0
    for bboxes in dict_elem["boxes"]:
      total_eqns += 1
      # crop from original iamge, and send that to MathPix
      x1, y1, x2, y2, _, label = bboxes

      # skip in-line equations (not skipping everything, but not sure if its correct)
      if label > 0.0:
        eqn_num += 1
        skipped_eqns += 1
        continue
      
      IMG_OUT_DIR = f"/tmp/cropped_imgs_{pdf_name}/"
      subprocess.run(["rm", "-rf", IMG_OUT_DIR])
      subprocess.run(["mkdir", "-p", IMG_OUT_DIR])

      crop_path = IMG_OUT_DIR + "_p"+ str(page_num) + "_e" + str(eqn_num) + ".png"
      page_png_path = f"/tmp/converted_pdfs_{pdf_name}/" + str(page_num) + ".png"
      model_width, model_height = 640,640
      image = Image.open(page_png_path).convert('RGB')
      width, height = image.size
      x_ratio, y_ratio = width/model_width, height/model_height

      # CROP original PNG with yolo bounding box coordinates
      x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
      y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
      cropped_image = image.crop((x1, y1, x2, y2))
      cropped_image.save(crop_path)
      
      latex_string = image_to_latex_convert(open(crop_path, "rb"), query_bool=False)
      latex_string = latex_string.replace(" ", "")
      print(f"{eqn_num} on {page_num}: {latex_string}")
      equations_list.append((latex_string, page_num, eqn_num))
      eqn_num += 1
    print(f"page {page_num}: skipped {skipped_eqns} in-line eqns, out of {total_eqns}.")
    
  print("Finished all MathPix API calls!")

  # sort equations by second element in tuple i.e. edit_dist_from_query
  # return equations with top_n smallest edit distances
  top_n = 5
  sorted_lst = levenshtein_distance(query_string=query_text, latex_list=equations_list, top_n=top_n)
  print("most similar eqns: ", sorted_lst)
  return sorted_lst

#print("Finished parse_tree_similarity")

def lambda_handler(event, context):
  try:
      print("Running backend...")
      handler = dataHandler.DataHandler()
      objects = handler.list_s3_objects("mathsearch-intermediary")

      body = json.loads(event['Records'][0]['body'])
      receipt_handle = event['Records'][0]['receiptHandle']
      file = body['Records'][0]['s3']['object']['key']
      print("File name: ", file)

      uuid = handler.extract_uuid(file)
      expected_image = f'{uuid}_image'

      if handler.is_expected_image_present(objects, expected_image):
          print('Found image, run ML model')
      
          # clear tmp folder before running the ML model
          subprocess.call('rm -rf /tmp/*', shell=True)

          # folders which we download S3 bucket PDF to
          png_converted_pdf_path = "/tmp/converted_pdfs"
          pdfs_from_bucket_path = "/tmp/pdfs_from_bucket"
          yolo_crops_path = "/tmp/crops/"

          # create the pdfs_from_bucket directory if it doesn't exist
          subprocess.run(f'mkdir -p {pdfs_from_bucket_path}', shell=True, cwd="/tmp")
          subprocess.run(f'mkdir -p {yolo_crops_path}', shell=True, cwd="/tmp")

          pdf_name = uuid+"_pdf"
          query_name = uuid+"_image"
          local_pdf, png_pdf_path, local_target = download_files(pdf_name, query_name, png_converted_pdf_path, pdfs_from_bucket_path)

          ## CALL TO LOCAL COMPUTE CLUSTER API TO RUN YOLO
          # Check if API is available
          try:
              health_response = requests.get(f"{ML_API_URL}/health", timeout=5)
              if health_response.status_code != 200:
                  raise Exception(f"API health check failed: {health_response.status_code}")
          except Exception as e:
              return {
                  'statusCode': 400,
                  'body': json.dumps('Error connecting to ML API'),
                  'error': str(f"Error connecting to ML API at {ML_API_URL}: {e}")
              }

          print(f"Sending to local ML API at {ML_API_URL}...")
          yolo_result = []
          os.chdir(png_converted_pdf_path+"_"+ pdf_name)
          infer_start_time = time.time()
          
          for file in os.listdir(png_pdf_path):
            # don't need to run inference on query.png
            if file == "query.png": continue

            print(f"Processing {file}")
            
            page_num = file.split(".")[0]
            file_path = os.path.join(png_pdf_path, file)
            
            # Send image file to API
            try:
                with open(file_path, 'rb') as f:
                    response = requests.post(
                        f"{ML_API_URL}/predict",
                        files={'file': ('page.png', f, 'image/png')},
                        timeout=60
                    )
                
                if response.status_code != 200:
                    print(f"Error processing {file}: {response.status_code} - {response.text}")
                    # Continue with empty result for this page
                    yolo_result.append(({"boxes": []}, page_num))
                    continue
                
                api_result = response.json()
                
                # Transform API response to match expected format
                # API returns: {"boxes": [{"bbox": [x1,y1,x2,y2], "confidence": score, "class": label, ...}], "count": N}
                # Code expects: {"boxes": [[x1, y1, x2, y2, confidence, label], ...]}
                transformed_boxes = []
                for box in api_result.get("boxes", []):
                    bbox = box["bbox"]  # [x1, y1, x2, y2]
                    confidence = box["confidence"]
                    class_label = box["class"]
                    # Format: [x1, y1, x2, y2, confidence, label]
                    transformed_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence, class_label])
                
                yolo_result.append(({"boxes": transformed_boxes}, page_num))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                # Continue with empty result for this page
                yolo_result.append(({"boxes": []}, page_num))
          
          infer_end_time = time.time()
          print(f"ML API Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")

          print("ML API results received!")
          if yolo_result:
              print(f"Length of ML API results: {len(yolo_result)} pages")
          print(yolo_result)

          top5_eqns = rank_eqn_similarity(yolo_result=yolo_result, query_path=local_target, pdf_name=pdf_name)
          print("MathPix API calls completed, and tree similarity generated!")

          page_nums_5 = sorted(set(([page_num for (latex_string, page_num, eqn_num, dist) in top5_eqns])))
          top5_eqns_info = [(page_num, eqn_num) for (latex_string, page_num, eqn_num, dist) in top5_eqns]
          #print("top_5_eqns_info ", top_5_eqns_info)

          # sort by page number
          bboxes_dict = {}
          for dict_elem, page_num in yolo_result:
            # don't draw bounding boxes on pages that don't have top 5 equation
            if page_num not in page_nums_5:
              continue
            count = 1
            for bboxes in dict_elem["boxes"]:
              # only collect bounding boxes from top 5 equation
              if (page_num, count) in top5_eqns_info:
                rank = top5_eqns_info.index((page_num, count))
                if page_num in bboxes_dict.keys():
                  bboxes_dict[page_num].append((bboxes[:4], rank))
                else:
                  bboxes_dict[page_num] = [(bboxes[:4], rank)]
              count += 1

          # draws the bounding boxes for the top 5 equations and converts pages back to PDF
          # final PDF with bounding boxes saved in directory pdf_out
          final_pdf_path = final_output(pdf_name, png_pdf_path, bboxes_dict)

          # return JSON with the following keys
          # id: UUID
          # pdf : path / pdf name
          # pages : list of page numbers sorted in order of most to least similar to query []
          # bbox: list of tuples (page_num, [list of equation label + four coordinates of bounding box])
          pages = [int(p)+1 for p in page_nums_5]
          json_result = {"statusCode" : 200, "body": "Successfully queried and processed your document!", 
                        "id": uuid, "pdf": pdf_name, "pages": pages, "bbox": bboxes}
            
          print(f"final json_result {json_result}")
      
      # Dequeue from SQS
      handler.delete_sqs_message(QUEUE_URL, receipt_handle)

      # Write json_result to local OUTPUT_DIR
      tmp_json_path = f"/tmp/{uuid}_results.json" 
      with open(tmp_json_path, "w") as outfile:
        json.dump(json_result, outfile)
    
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      final_json_path = os.path.join(OUTPUT_DIR, f"{uuid}_results.json")
      shutil.copy2(tmp_json_path, final_json_path)

      return json_result
       
  except:
    exception = traceback.format_exc()
    print(f"Error: {exception}")
    return {
        'statusCode': 400,
        'body': json.dumps(f'Error processing the document.'),
        'error': exception
    }

#Replacement function for lambda_handler/SQS
def process_local_job(uuid, input_dir, output_dir):
  try:
      print(f"--- Processing {uuid} ---")

      # Clean tmp
      subprocess.call('rm -rf /tmp/*', shell=True)

      png_converted_pdf_path = "/tmp/converted_pdfs"
      pdfs_from_bucket_path = "/tmp/pdfs_from_bucket"
      yolo_crops_path = "/tmp/crops/"
      subprocess.run(f'mkdir -p {pdfs_from_bucket_path}', shell=True, cwd="/tmp")
      subprocess.run(f'mkdir -p {yolo_crops_path}', shell=True, cwd="/tmp")

      # 1. Local Download
      pdf_name = uuid + "_pdf"
      query_name = uuid + "_image"
      local_pdf, png_pdf_path, local_target = download_files(pdf_name, query_name, png_converted_pdf_path, pdfs_from_bucket_path)

      # 2. Local ML API Call
      # Check if API is available
      try:
          health_response = requests.get(f"{ML_API_URL}/health", timeout=5)
          if health_response.status_code != 200:
              raise Exception(f"API health check failed: {health_response.status_code}")
      except Exception as e:
          print(f"Error connecting to ML API at {ML_API_URL}: {e}")
          with open(os.path.join(output_dir, f"{uuid}_result.json"), "w") as outfile: 
              json.dump({"status": "error", "error": f"Error connecting to ML API: {e}"}, outfile)
          return False

      print(f"Sending to local ML API at {ML_API_URL}...")
      yolo_result = []
      
      # Standardize file sorting
      files = sorted(os.listdir(png_pdf_path), key=lambda x: int(x.split('.')[0]) if x.replace('.','').isdigit() else 0)
      
      for file in files:
        if file == "query.png" or not file.endswith(".png"): continue

        print(f"Processing {file}")
        page_num = file.split(".")[0]
        file_path = os.path.join(png_pdf_path, file)
        
        # Send image file to API
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{ML_API_URL}/predict",
                    files={'file': ('page.png', f, 'image/png')},
                    timeout=60
                )
            
            if response.status_code != 200:
                print(f"Error processing {file}: {response.status_code} - {response.text}")
                # Continue with empty result for this page
                yolo_result.append(({"boxes": []}, page_num))
                continue
            
            api_result = response.json()
            
            # Transform API response to match expected format
            # API returns: {"boxes": [{"bbox": [x1,y1,x2,y2], "confidence": score, "class": label, ...}], "count": N}
            # Code expects: {"boxes": [[x1, y1, x2, y2, confidence, label], ...]}
            transformed_boxes = []
            for box in api_result.get("boxes", []):
                bbox = box["bbox"]  # [x1, y1, x2, y2]
                confidence = box["confidence"]
                class_label = box["class"]
                # Format: [x1, y1, x2, y2, confidence, label]
                transformed_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence, class_label])
            
            yolo_result.append(({"boxes": transformed_boxes}, page_num))
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            # Continue with empty result for this page
            yolo_result.append(({"boxes": []}, page_num))

      print("ML API results received!")
      top5_eqns = rank_eqn_similarity(yolo_result=yolo_result, query_path=local_target, pdf_name=pdf_name)

      page_nums_5 = sorted(set(([page_num for (latex_string, page_num, eqn_num, dist) in top5_eqns])))
      top5_eqns_info = [(page_num, eqn_num) for (latex_string, page_num, eqn_num, dist) in top5_eqns]

      bboxes_dict = {}
      for dict_elem, page_num in yolo_result:
        if page_num not in page_nums_5: continue
        count = 1
        for bboxes in dict_elem["boxes"]:
          if (page_num, count) in top5_eqns_info:
            rank = top5_eqns_info.index((page_num, count))
            if page_num in bboxes_dict.keys():
              bboxes_dict[page_num].append((bboxes[:4], rank))
            else:
              bboxes_dict[page_num] = [(bboxes[:4], rank)]
          count += 1

      # 3. Generate Output
      final_pdf_path = final_output(pdf_name, png_pdf_path, bboxes_dict)

      # 4. Generate JSON
      pages = [int(p)+1 for p in page_nums_5]
      json_result = {
          "id": uuid, 
          "status": "success",
          "pdf_name": f"{uuid}.pdf", 
          "pages": pages, 
          "bbox": bboxes_dict
      }
        
      with open(os.path.join(output_dir, f"{uuid}_result.json"), "w") as outfile: 
        json.dump(json_result, outfile)

      return True
       
  except:
    exception = traceback.format_exc()
    print(f"Error: {exception}")
    # Write error JSON to output so worker knows we failed
    with open(os.path.join(output_dir, f"{uuid}_result.json"), "w") as outfile: 
        json.dump({"status": "error", "error": exception}, outfile)
    return False
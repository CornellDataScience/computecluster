import React, { useState, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "./Results.css";
import NavBar from "../NavBar.js";
import { useParams } from "react-router-dom";

const SERVER = "https://staticservermathsearch.cornelldata.science";

// 1. WORKER CONFIGURATION
// Use explicit HTTPS and unpkg for better stability with version 2.16.105
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@2.16.105/build/pdf.worker.min.js`;

const Results = () => {
  const routeParams = useParams();
  const uuid = routeParams.uuid;

  // We will store the "blob:..." string URL here, not the raw data
  const [pdfUrl, setPdfUrl] = useState(null);
  const [pages, setPages] = useState([]);
  
  const [loading, setLoading] = useState(true);
  const [pdfDownloaded, setPdfDownloaded] = useState(false);
  const [jsonDownloaded, setJsonDownloaded] = useState(false);
  const [numPages, setNumPages] = useState(null);

  const downloadRequest = async (uuid) => {
    try {
      // --- PDF DOWNLOAD ---
      if (!pdfDownloaded) {
        const pdfResponse = await fetch(`${SERVER}/results/${uuid}.pdf`);
        
        if (pdfResponse.ok) {
          const pdfBlob = await pdfResponse.blob();

          if (pdfBlob.size > 0) {
            // 1. HEAD CHECK: Verify it starts with %PDF
            const header = await pdfBlob.slice(0, 5).text();
            
            // 2. TAIL CHECK: Verify it ends with %%EOF
            const tail = await pdfBlob.slice(pdfBlob.size - 100, pdfBlob.size).text();

            if (header === "%PDF-" && tail.includes("%%EOF")) {
               console.log("PDF Validated (Header & EOF found)");
               
               // 3. CREATE BLOB URL
               // This creates a temporary local URL (string) that is safer to pass to the Worker
               const objectUrl = URL.createObjectURL(pdfBlob);
               setPdfUrl(objectUrl); 
               setPdfDownloaded(true);
            } else {
               console.warn("PDF incomplete or invalid. Retrying...", { header, tailEnd: tail.slice(-20) });
            }
          }
        }
      }

      // --- JSON DOWNLOAD ---
      if (!jsonDownloaded) {
        const jsonResponse = await fetch(`${SERVER}/results/${uuid}_result.json`);
        
        if (jsonResponse.ok) {
          const jsonText = await jsonResponse.text();
          if (jsonText.length > 0) {
             try {
                const json = JSON.parse(jsonText);
                if (json.pages && json.pages.length > 0) {
                   setPages(json.pages);
                   setJsonDownloaded(true);
                }
             } catch (e) {
                console.warn("Invalid JSON structure");
             }
          }
        }
      }

    } catch (err) {
      console.error("Error downloading:", err);
    }
  };

  // EFFECT 1: Stop loading ONLY when both files are valid
  useEffect(() => {
    if (pdfDownloaded && jsonDownloaded) {
      setLoading(false);
    }
  }, [pdfDownloaded, jsonDownloaded]);

  // EFFECT 2: Polling Loop
  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        downloadRequest(uuid);
      }, 2000);

      downloadRequest(uuid); 

      return () => clearInterval(interval);
    }
  }, [loading, uuid, pdfDownloaded, jsonDownloaded]);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
  }

  const renderPages = () => {
    const pdfPages = [];
    for (let i = 1; i <= numPages; i++) {
      pdfPages.push(
        <Page
          key={i}
          renderAnnotationLayer={false}
          renderTextLayer={false}
          pageNumber={i}
        />
      );
    }
    return pdfPages;
  };

  const scroll = (id) => {
    const target = document.getElementById(id);
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <>
      {loading ? (
        <div className="page">
          <div className="center">
            <div className="loader"></div>
            <p style={{marginTop: "20px"}}>Processing...</p>
          </div>
        </div>
      ) : (
        <div style={{ backgroundColor: "#eeeeee" }}>
          <NavBar />
          {pdfUrl && pages.length > 0 && (
            <div>
              <br />
              <div className="grid-container">
                <div></div>

                {/* PDF Viewer */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    height: "90vh",
                    overflow: "auto",
                  }}
                >
                  <Document
                    file={pdfUrl} // Pass the Blob URL string directly
                    onLoadSuccess={onDocumentLoadSuccess}
                    onLoadError={(error) => console.error("PDF Load Error:", error)}
                  >
                    {renderPages().map((item, index) => (
                      <div id={index + 1} key={index}>
                        {item}
                        <br />
                      </div>
                    ))}
                  </Document>
                </div>

                {/* Nav buttons */}
                <div
                  style={{
                    background: "white",
                    borderRadius: "1em",
                    display: "flex",
                    justifyContent: "center",
                  }}
                >
                  <div style={{ padding: "1em" }}>
                    <div style={{ textAlign: "center", paddingBottom: "2em" }}>
                      <b>Results</b>
                    </div>
                    {pages.map((item, index) => (
                      <div key={index}>
                        <button className="button" onClick={() => scroll(item)}>
                          Page {item}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default Results;
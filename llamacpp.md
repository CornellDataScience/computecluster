type in `llamacpphelp` for the info below whenever you need it.

llamacppchat to run inference and open the default chat window

llamacppprompt to only ask one prompt (one-and-done)

llamacppwarm will load the model quietly in the background so it runs faster when you call it

--jinja to use a jinja template  (cleaner format)

-p to just input one prompt. example: -p solve P=NP

-i for interactive mode, will allow you to send multiple prompts

-n  to determine output tokens; default is -1 which is unlimited or EOS. example: -n 256 for 265 tokens

-c for context window sizing. example: -c 2048 for 2048 context

-mli for multiple lines of input, good for long prompts

-t for the number of CPU threads, best if number of cores but should not matter too much if using a GPU. example: -t 8 for 8 threads

-ngl for number of GPU layers. example: -ngl 999 for maximum layers

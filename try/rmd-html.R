args<-commandArgs(trailingOnly = TRUE)
if(grepl("\\.rmd$", args[1], ignore.case=T)) args[1] <- substr(args[1], 1, nchar(args[1])-4)
if (file.exists(paste(args[1], ".rmd", sep=""))) 
{library("knitr")

# New processing functions
process_tangle <- function (x) { 
    UseMethod("process_tangle", x)
}

process_tangle.block <- function (x) {
    params = opts_chunk$merge(x$params)

    # Suppress any code but python
    if (params$engine != 'R') {
        params$purl <- FALSE
    }
    if (isFALSE(params$purl)) 
        return("")
    label = params$label
    ev = params$eval
    code = if (!isFALSE(ev) && !is.null(params$child)) {
               cmds = lapply(sc_split(params$child), knit_child)
               one_string(unlist(cmds))
           }
           else knit_code$get(label)
    if (!isFALSE(ev) && length(code) && any(grepl("read_chunk\\(.+\\)", 
                                                  code))) {
        eval(parse_only(unlist(stringr::str_extract_all(code, 
                                                        "read_chunk\\(([^)]+)\\)"))))
    }
    code = knitr:::parse_chunk(code)
    if (isFALSE(ev)) 
        code = knitr:::comment_out(code, params$comment, newline = FALSE)
                                        # Output only the code, no documentation
    return(knitr:::one_string(code))
}

                                        # Reassign functions
    utils::assignInNamespace("process_tangle.block",
                             process_tangle.block,
                             ns="knitr")
    
    knitr::purl(input=paste(args[1], ".rmd", sep=""), output=paste(args[1], ".R", sep=""), encoding="utf-8")

# New processing functions
process_tangle <- function (x) { 
    UseMethod("process_tangle", x)
}

process_tangle.block <- function (x) {
    params = opts_chunk$merge(x$params)

    # Suppress any code but python
    if (params$engine != 'python') {
        params$purl <- FALSE
    }
    if (isFALSE(params$purl)) 
        return("")
    label = params$label
    ev = params$eval
    code = if (!isFALSE(ev) && !is.null(params$child)) {
               cmds = lapply(sc_split(params$child), knit_child)
               one_string(unlist(cmds))
           }
           else knit_code$get(label)
    if (!isFALSE(ev) && length(code) && any(grepl("read_chunk\\(.+\\)", 
                                                  code))) {
        eval(parse_only(unlist(stringr::str_extract_all(code, 
                                                        "read_chunk\\(([^)]+)\\)"))))
    }
    code = knitr:::parse_chunk(code)
    if (isFALSE(ev)) 
        code = knitr:::comment_out(code, params$comment, newline = FALSE)
                                        # Output only the code, no documentation
    return(knitr:::one_string(code))
}

                                        # Reassign functions
    utils::assignInNamespace("process_tangle.block",
                             process_tangle.block,
                             ns="knitr")
    
    knitr::purl(input=paste(args[1], ".rmd", sep=""), output=paste(args[1], ".py", sep=""), encoding="utf-8")

    rmarkdown::render(input=paste(args[1], ".rmd", sep=""), 
                      "bookdown::html_document2",
                      ## 使用html_document()函数使得 yaml 前言无法产生作用
                      ## rmarkdown::html_document(keep_md=TRUE, number_sections = T, css = "css/markdown.css"),
                      encoding="utf-8")
}else{
    rmarkdown::render(input=paste(args[1], ".md", sep=""), "bookdown::html_document2",encoding="utf-8")
}

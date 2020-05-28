@echo off
set _SCRIPT_PATH=%~dp0

Rscript %_SCRIPT_PATH%rmd-html.R %1

set var=%1

if exist %var%.html (start %var%.html) else (start %var:~0,-4%.html)

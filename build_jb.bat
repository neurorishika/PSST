CALL copy Book\_config.yml _config.yml /y
CALL copy Book\_toc.yml _toc.yml /y
CALL conda activate jupyterbook
CALL cd ..
CALL jupyter-book build PSST/
CALL cd PSST
CALL ghp-import -n -p -f _build/html
CALL del _config.yml
CALL del _toc.yml
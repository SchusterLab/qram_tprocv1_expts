@echo off
echo Preparing to copy from %1 to %cd%.
set /p=Press any key to continue...
call copy %1\*.ipynb .
call copy %1\*.yml .
call copy %1\experiments\*.py .\experiments
call copy %1\experiments\single_qubit\* .\experiments\single_qubit
call copy %1\experiments\two_qubit\* .\experiments\two_qubit
call copy %1\experiments\three_qubit\* .\experiments\three_qubit
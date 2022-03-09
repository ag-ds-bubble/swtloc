#!/bin/bash
. ../../variables.sh

# All the supported python versions
declare -a pyversions=("3.6" "3.7" "3.8" "3.9" "3.10")

# shellcheck disable=SC2154
venvs_path="${root_path}venvs/"
project_rootpath="${root_path}swtloc/"
dist_path="${root_path}swtloc/dist/*"

logs_path="${root_path}swtloc/logs/"
test_log_file="${root_path}swtloc/logs/tests_log.txt"
upload_log_file="${root_path}swtloc/logs/upload.txt"
rm -r "$logs_path"
mkdir "$logs_path"
# shellcheck disable=SC2164
cd "$logs_path"
touch "$test_log_file"
touch "$upload_log_file"

# shellcheck disable=SC2164
cd "$project_rootpath"

# shellcheck disable=SC2164
cd "$project_rootpath"
echo -e 'Uploading to TestPyPI & Download to Venv + Rerun of the tests:'
for i in "${pyversions[@]}"
do
  echo -e '\t For Python =' "$i"

  echo -e '\t\tBuilding ...'
  {
    echo '=============================================Python = v' "$i"'============================================='
    echo '-----------------------------------------------BUILDING-----------------------------------------------'
    # Upload to TestPyPi, download and then test again for the
    pydv="${venvs_path}py${i//.}DevVenv"
    # shellcheck disable=SC1090
    # shellcheck disable=SC2154
    source "$conda_sh"
    conda activate "$pydv"
    echo -e '\t\tDevelopment Venv : ' "$(which python)"
    # Upload to TestPyPI
    python setup_dev.py sdist
    python setup_dev.py bdist_wheel
    } >> "$upload_log_file" 2>&1

  echo -e '\t\tUploading ...'
  {
    echo '-----------------------------------------------UPLOADING-----------------------------------------------'
    # shellcheck disable=SC2154
    python -m twine upload --verbose -u="$uname" -p="$psw" -r testpypi "$dist_path"
    conda deactivate
    sleep 5  # So that the fresh pull is made
    } >> "$upload_log_file" 2>&1

  echo -e '\t\tDownloading ...'
  {
    echo '-----------------------------------------------DOWNLOADING-----------------------------------------------'
    # Build the wheel file & Test for the Dev Environment
    # Remove the build files
    rm -r build
    rm -r dist
    rm -r swtloc.egg-info
    # Activate the Venv
    pdv="${venvs_path}py${i//.}Venv"
    conda activate "$pdv"
    echo -e '\t\tVenv : ' "$(which python)"
    pip install --no-cache-dir -I --extra-index-url https://test.pypi.org/simple/ swtloc
    pip install pickle5
    } >> "$upload_log_file" 2>&1

  echo -e '\t\tRunning Tests ...'
  {
    echo '=============================================Python = v' "$i"'============================================='
    echo '-----------------------------------------------RUNNING TESTS-----------------------------------------------'
    python -m unittest
    conda deactivate
    echo '-----------------------------------------------=============-----------------------------------------------'
    } >> "$test_log_file" 2>&1
  echo -e '\t Done!'
  sleep 5  # Precautionary

done

#!/bin/bash
. "${PWD}/variables.sh"


# All the supported python versions
declare -a pyversions=("3.6")

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
  # Remove the build files
  rm -r build
  rm -r dist
  rm -r swtloc.egg-info
  echo -e '\t For Python =' "$i"
  pydv="${venvs_path}py${i//.}TestVenv"
  rm -r "$pydv"

  echo -e '\t\tBuilding Venv & distributions...'
  {
    echo '=============================================Python = v' "$i"'============================================='
    source "$conda_sh"
    echo '---------------------------------------------VENV-CREATE----------------------------------------------'
    conda create -p="$pydv" python="$i" -y
    conda activate "$pydv"

    echo '-----------------------------------------------BUILDING-----------------------------------------------'
    # shellcheck disable=SC1090
    # shellcheck disable=SC2154
#    source "D:/softs/anaconda3/etc/profile.d/conda.sh"
    conda activate "$pydv"
    echo -e '\t\tTesting Venv : ' "$(which python)"
    conda install twine -y
    conda install pickle5 -y

    python setup_dev.py sdist
    python setup_dev.py bdist_wheel
    } >> "$upload_log_file" 2>&1

  echo -e '\t\tUploading ...'
  {
    echo '-----------------------------------------------UPLOADING-----------------------------------------------'
    # shellcheck disable=SC2154
    # Upload to TestPyPi, download and then test again for the
    python -m twine upload --verbose --skip-existing -u="$uname" -p="$psw" -r testpypi "$dist_path"
    conda deactivate
    } >> "$upload_log_file" 2>&1

  echo -e '\t\tDownloading ...'
  {
    sleep 10 # Precautionary
    echo '-----------------------------------------------DOWNLOADING-----------------------------------------------'
    conda activate "$pydv"
    echo -e '\t\tTestVenv : ' "$(which python)"
    pip install --no-cache-dir -I --extra-index-url https://test.pypi.org/simple/ swtloc
#    conda install 'numpy=1.21' -y
    } >> "$upload_log_file" 2>&1

  echo -e '\t\tRunning Tests ...'
  {
    echo '=============================================Python = v' "$i"'============================================='
    echo '-----------------------------------------------RUNNING TESTS-----------------------------------------------'
    pip list
    python -m unittest
    conda deactivate
    echo '-----------------------------------------------=============-----------------------------------------------'
    } >> "$test_log_file" 2>&1
  echo -e '\t Done!'
  sleep 100 # Precautionary

done

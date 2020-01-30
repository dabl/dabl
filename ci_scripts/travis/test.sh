set -e

# check if we do not leave artifacts
mkdir -p $TEST_DIR
cp setup.cfg $TEST_DIR

cd $TEST_DIR

pytest --pyargs dabl

cd -
pytest doc

set -e

# check if we do not leave artifacts
mkdir -p $TEST_DIR

cd $TEST_DIR


pytest --pyargs fml


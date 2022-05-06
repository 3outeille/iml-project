# Instructions to use evaluate student submissions using Docker

We will run student code inside a Docker container, to ensure minimal safety.
Evaluation code will be run outside the container.

## File paths

- Student code (process.py and other required files) will be mounted on `/app`.
- The test set (no ground truth file) will be mounted on `/data`
- Output data will go to `/output/result.csv`

## Container image
We will build the image using the `Dockerfile` provided.

It is a simple Python3.8-slim image with minimal requirements.

Build the image with this command:
```shell
cd docker
docker build -t jchazalon/imleval .
```

## Invocation process
Student code will be invoked with the following command:
```shell
docker run --rm -it \
    -v ${PATH_TO_TEST_SET}:/data:ro \
    -v ${PATH_TO_STUDENT_CODE}:/app:ro \
    -v ${PATH_TO_STUDENT_OUTPUT_DIR}:/output/ \
    --network none \
    --memory 500m \
    --memory-swap 500m \
    jchazalon/imleval \
    python3 process.py --test_dir /data --output /output/result.csv
```

## Evaluation
Run the evaluation with the following command:
```shell
python autoeval.py --ground_truth ${PATH_TO_TEST_SET_GT}/gt.txt --submission ${PATH_TO_STUDENT_OUTPUT_DIR}/result.csv
```

This should display something like this for random results:
```
Score:  3.57%; c=2; r=0; e=54; N=56
```

Check `autoeval.py`'s code for more details.

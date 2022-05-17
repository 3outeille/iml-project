PATH_TO_TEST_SET = ${PWD}/data
PATH_TO_STUDENT_CODE = ${PWD}/app
PATH_TO_STUDENT_OUTPUT_DIR = ${PWD}/output
PATH_TO_TEST_SET_GT = ${PWD}

build:
	docker build -t jchazalon/imleval .

predict:
	docker run --rm -it \
		-v ${PATH_TO_TEST_SET}:/data:ro \
		-v ${PATH_TO_STUDENT_CODE}:/app:ro \
		-v ${PATH_TO_STUDENT_OUTPUT_DIR}:/output/ \
		--network none \
		--memory 500m \
		--memory-swap 500m \
		jchazalon/imleval \
		python3 process.py --test_dir /data --output /output/result.csv

submission:
	python autoeval.py --ground_truth ${PATH_TO_TEST_SET_GT}/gt.txt --submission ${PATH_TO_STUDENT_OUTPUT_DIR}/result.csv

clean:
	rm output/*.csv data/*  

PHONY: clean

submission:
	python src/main.py
	python rendu/process.py  --test_dir=rendu/test_fake --output=result.csv
	python rendu/autoeval.py --ground_truth rendu/test_fake/gt.txt --submission result.csv
clean:
	rm -rf *-results *.egg-info assets/*.pkl assets/*.csv *.csv

PHONY: clean

.PHONY: install all test serve docker-build docker-up

install:
	pip install -r requirements.txt

all:            ## download data, clean, train, evaluate, write artifacts and reports
	python -m inactivity all

test:
	python -m pytest -q

serve:
	python -m inactivity serve

docker-build:
	docker build -t inactivity-api:0.1.0 .

docker-up:
	docker compose up --build

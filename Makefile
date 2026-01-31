.PHONY: gen up down logs

gen:
	protoc -I proto \
	  --go_out=ingestor-go/pb --go_opt=paths=source_relative \
	  --go-grpc_out=ingestor-go/pb --go-grpc_opt=paths=source_relative \
	  proto/video.proto
	python -m grpc_tools.protoc -I proto \
	  --python_out=client-webcam \
	  --grpc_python_out=client-webcam \
	  proto/video.proto

up:
	cd infra && docker compose up -d --build

down:
	cd infra && docker compose down -v

logs:
	cd infra && docker compose logs -f --tail=200

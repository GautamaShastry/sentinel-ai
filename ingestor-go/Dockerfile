FROM golang:1.22 AS build
WORKDIR /app
COPY . .
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o /bin/ingestor ./cmd/ingestor

FROM gcr.io/distroless/base-debian12
COPY --from=build /bin/ingestor /ingestor
EXPOSE 50051 9100
ENTRYPOINT ["/ingestor"]

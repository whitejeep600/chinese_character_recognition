docker build --rm -t recognition_image .
docker container run -t -d --name recognition_container recognition_image


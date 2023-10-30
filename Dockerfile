# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /microsom

# Install curl
RUN apt-get update && apt-get install -y curl

# Copy the current directory contents into the container at /app
COPY . /microsom

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN python ./src/train.py

# Make port 80 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "./app/app.py"]
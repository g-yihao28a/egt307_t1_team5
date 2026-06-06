# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory to match your compose file
WORKDIR /work

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of project code
COPY . .

# Command to run pipeline
CMD ["bash", "run.sh"]
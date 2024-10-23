# Using Python runtime as a parent image
FROM python:3.9-slim

# Setting the working directory in the container
WORKDIR /Multiple Disease Prediction webApp 

# Copy the current directory contents into the container at /app
COPY . /Multiple disease prediction model/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for streamlit
EXPOSE 8501

# Command to run the streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]


#docker build -t disease-prediction-app         - image build command
#docker run -p 8501:8501 disease-prediction-app             -image run command
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import List
from pydantic import BaseModel
import pandas as pd
import os
import logging
from zoneinfo import ZoneInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sensor Data API",
    description="Real-time sensor data monitoring API",
    version="1.0.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins in production (e.g., https://your-app.onrender.com)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the root directory
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Data model
class SensorData(BaseModel):
    timestamp: str
    speed: float
    voltage: float
    current: float
    soc: float
    temperature: float  # Renamed from 'temp'
    pdutemp: float
    power: float
    lat: float
    long: float
    num_of_sats: float
    wind_speed: float
    wind_direction: float  # Using wind_dir_deg
    wind_level: float
    wind_relative_angle: float

class SensorDataResponse(BaseModel):
    data: List[SensorData]
    count: int
    latest_timestamp: str

# CSV file path (in root directory)
CSV_FILE_PATH = "final_clean_data.csv"

def load_csv_data():
    try:
        logger.info(f"Loading all rows from CSV: {CSV_FILE_PATH}")
        if not os.path.exists(CSV_FILE_PATH):
            logger.error(f"CSV file not found at {CSV_FILE_PATH}")
            raise FileNotFoundError(f"CSV file {CSV_FILE_PATH} not found")

        # Required columns based on your CSV structure
        required_columns = [
            'timestamp', 'speed', 'voltage', 'current', 'soc', 'temp', 'pdutemp',
            'power', 'lat', 'long', 'num_of_sats', 'wind_speed',
            'wind_dir_deg', 'wind_level', 'wind_relative_angle'
        ]

        # Load the entire CSV at once for better performance with large datasets
        logger.info("Reading CSV file...")
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"Initial CSV load: {len(df)} rows")

        # Verify required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert Unix timestamp to IST
        logger.info("Converting timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')

        # Rename columns to match the expected format
        df = df.rename(columns={'temp': 'temperature', 'wind_dir_deg': 'wind_direction'})

        # Clean and validate data
        logger.info("Cleaning and validating data...")
        
        # Remove rows with any NaN values in critical columns
        numeric_columns = [
            'speed', 'voltage', 'current', 'soc', 'temperature',
            'pdutemp', 'power', 'lat', 'long', 'num_of_sats',
            'wind_speed', 'wind_direction', 'wind_level', 'wind_relative_angle'
        ]
        
        # Count initial rows
        initial_count = len(df)
        
        # Drop rows with NaN values
        df = df.dropna(subset=numeric_columns)
        
        logger.info(f"After removing NaN values: {len(df)} rows (removed {initial_count - len(df)} rows)")

        # Ensure all numeric columns are properly typed
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows that couldn't be converted to numeric
        df = df.dropna(subset=numeric_columns)
        logger.info(f"After numeric conversion: {len(df)} rows")

        if len(df) == 0:
            logger.error("No valid rows found in CSV after cleaning")
            raise ValueError("No valid data rows found in CSV after cleaning")

        # Convert to list of dictionaries
        logger.info("Converting to list format...")
        valid_data = []
        for _, row in df.iterrows():
            try:
                row_dict = {
                    'timestamp': row['timestamp'].isoformat(),
                    'speed': float(row['speed']),
                    'voltage': float(row['voltage']),
                    'current': float(row['current']),
                    'soc': float(row['soc']),
                    'temperature': float(row['temperature']),
                    'pdutemp': float(row['pdutemp']),
                    'power': float(row['power']),
                    'lat': float(row['lat']),
                    'long': float(row['long']),
                    'num_of_sats': float(row['num_of_sats']),
                    'wind_speed': float(row['wind_speed']),
                    'wind_direction': float(row['wind_direction']),
                    'wind_level': float(row['wind_level']),
                    'wind_relative_angle': float(row['wind_relative_angle'])
                }
                valid_data.append(row_dict)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue

        if not valid_data:
            logger.error("No valid rows found after processing")
            raise ValueError("No valid data rows found after processing")

        # Sort by timestamp (ascending for proper time series)
        valid_data.sort(key=lambda x: datetime.fromisoformat(x['timestamp'].replace('Z', '+00:00')))
        logger.info(f"Returning {len(valid_data)} valid rows, sorted by timestamp")
        return valid_data
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Sensor Data Monitoring API",
        "version": "1.0.0",
        "endpoints": {
            "/data": "Get all sensor data from predefined CSV",
            "/health": "Health check"
        }
    }

@app.get("/data", response_model=SensorDataResponse)
async def get_sensor_data():
    try:
        logger.info("Received request for /data")
        data = load_csv_data()
        if not data:
            logger.error("No valid data available")
            raise HTTPException(status_code=500, detail="No valid data available")
        logger.info(f"Returning {len(data)} rows of data")
        return SensorDataResponse(
            data=data,
            count=len(data),
            latest_timestamp=data[-1]["timestamp"] if data else ""  # Latest is last in sorted array
        )
    except Exception as e:
        logger.error(f"Error retrieving sensor data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sensor data: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "active"
    }

if __name__ == "__main__":
    import uvicorn
    # Use environment variable for port to support Render; default to 8000 for local testing
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

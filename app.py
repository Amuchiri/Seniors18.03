from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
import os
from sqlalchemy import Column, Integer, String, Boolean, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from dotenv import load_dotenv
from gtts import gTTS
import sounddevice as sd  # ✅ Use sounddevice instead


# ✅ Load environment variables
load_dotenv()

# ✅ Configure PostgreSQL Connection
DATABASE_URL2 = os.getenv("DATABASE_URL2","postgresql://postgres:RWLqmWmssShysuCYNKGAhELnuEhzGypD@postgres.railway.internal:5432/railway")
if not DATABASE_URL2:
    raise ValueError("❌ Missing DATABASE_URL2 environment variable")

# ✅ Create PostgreSQL Engine
engine = create_engine(DATABASE_URL2, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()

# ✅ Function to Get Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ User Data Model (Includes Preferences)
class UserData(Base):
    __tablename__ = "user_data"

    id = Column(Integer, primary_key=True, index=True)
    age_group = Column(String)
    education_level = Column(String)
    income_level = Column(String)
    internet_access = Column(String)
    device_usage = Column(String)
    support_required = Column(String)
    egov_service_usage = Column(String)
    prior_digital_training = Column(String)
    household_tech_access = Column(String)
    digital_skills_level = Column(String)  # Predicted Literacy Level
    font_size = Column(String, default="default")  # ✅ Store user font choice
    high_contrast = Column(Boolean, default=False)  # ✅ Store high contrast mode
    created_at = Column(String, default="CURRENT_TIMESTAMP")
    queries = relationship("UserQuery", back_populates="user")

# ✅ User Queries Model
class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_data.id"))
    user_query = Column(String)
    language = Column(String)
    user = relationship("UserData", back_populates="queries")

# ✅ Create Tables
Base.metadata.create_all(bind=engine)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# ✅ Load Machine Learning Model
MODEL_PATH = os.path.join(os.getcwd(), "models", "digital_literacy_model5.pkl")
ENCODERS_PATH = os.path.join(os.getcwd(), "models", "label_encoders5.pkl")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

# ✅ Mistral AI Configuration

import os
import requests

MISTRAL_KEY = os.getenv("MISTRAL_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_KEY:
    raise ValueError("❌ `MISTRAL_KEY` is missing! Check Railway Variables.")

print(f"🔍 Using Mistral API Key (masked): {MISTRAL_KEY[:5]}*****")  # Masked for security

headers = {
    "Authorization": f"Bearer {MISTRAL_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "mistral-tiny",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
}

response = requests.post(MISTRAL_URL, headers=headers, json=payload)

if response.status_code == 200:
    print("✅ Mistral API Response:", response.json())
elif response.status_code == 401:
    print("❌ Error 401: Unauthorized - Your FastAPI app is not sending the key correctly.")
else:
    print(f"❌ Error {response.status_code}:", response.text)


def generate_learning_content(user_query, literacy_level, language):
    """ Generate AI response using Mistral API """
    
    prompt = f"""
    User Literacy Level: {literacy_level}
    Language: {language}
    User Question: {user_query}
    Provide a structured response with bullet points and easy-to-read formatting.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        print(f"🔍 Sending request to Mistral API: {data}")  # ✅ Debugging
        response = requests.post(MISTRAL_URL, json=data, headers=headers)
        response.raise_for_status()  # ✅ Raise error for bad responses

        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        #print(f"✅ Mistral API Response: {ai_response}")  # ✅ Debugging

        return ai_response if ai_response else "⚠️ No AI response generated."
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Mistral API Error: {e}")
        return "⚠️ Mistral AI is currently unavailable."

# ✅ Function to preprocess user data
def preprocess_new_data(user_data):
    import pandas as pd
    df = pd.DataFrame([user_data])

    categorical_columns = [
        "Age_Group", "Education_Level", "Income_Level", "Internet_Access",
        "Device_Usage", "Support_Required", "eGov_Service_Usage",
        "Prior_Digital_Training", "Household_Tech_Access"
    ]

    for col in categorical_columns:
        df[col] = df[col].astype(str).fillna("None")

        if "None" not in label_encoders[col].classes_:
            label_encoders[col].classes_ = list(label_encoders[col].classes_) + ["None"]

        if df[col].iloc[0] not in label_encoders[col].classes_:
            df[col] = "None"

        df[col] = label_encoders[col].transform(df[col])

    return df

# ✅ Predict Digital Literacy Level & Save User
@app.post("/predict/")
async def predict(
    request: Request,
    Age_Group: str = Form(...),
    Education_Level: str = Form(...),
    Income_Level: str = Form(...),
    Internet_Access: str = Form(...),
    Device_Usage: str = Form(...),
    Support_Required: str = Form(...),
    eGov_Service_Usage: str = Form(...),
    Prior_Digital_Training: str = Form(...),
    Household_Tech_Access: str = Form(...),
    db: Session = Depends(get_db)
):
    """ Predict digital literacy level & save user data """

    user_data = {
        "Age_Group": Age_Group,
        "Education_Level": Education_Level,
        "Income_Level": Income_Level,
        "Internet_Access": Internet_Access,
        "Device_Usage": Device_Usage,
        "Support_Required": Support_Required,
        "eGov_Service_Usage": eGov_Service_Usage,
        "Prior_Digital_Training": Prior_Digital_Training,
        "Household_Tech_Access": Household_Tech_Access
    }

    # ✅ Preprocess and predict literacy level
    X_new = preprocess_new_data(user_data)
    prediction = model.predict(X_new)[0]
    predicted_label = label_encoders["Digital_Skills_Level"].inverse_transform([prediction])[0]

    # ✅ Save user data to DB
    new_user = UserData(
        age_group=Age_Group,
        education_level=Education_Level,
        income_level=Income_Level,
        internet_access=Internet_Access,
        device_usage=Device_Usage,
        support_required=Support_Required,
        egov_service_usage=eGov_Service_Usage,
        prior_digital_training=Prior_Digital_Training,
        household_tech_access=Household_Tech_Access,
        digital_skills_level=predicted_label
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # ✅ Ensure `user_id` is available

    print(f"✅ New User Created: {new_user.id}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_level": predicted_label,
        "user_id": new_user.id  # ✅ Send `user_id` to frontend
    })

# ✅ Save User Preferences (Font Size & High Contrast)
@app.post("/save-preference/")
async def save_preference(
    user_id: int = Form(...),
    font_size: str = Form(...),
    high_contrast: bool = Form(...),
    db: Session = Depends(get_db)
):
    """Save user UI preferences (Font Size & High Contrast Mode)."""

    if user_id <= 0:
        return JSONResponse(content={"error": "Invalid user_id"}, status_code=400)

    user = db.query(UserData).filter(UserData.id == user_id).first()
    if not user:
        return JSONResponse(content={"error": "User not found!"}, status_code=404)

    user.font_size = font_size
    user.high_contrast = high_contrast
    db.commit()

    return JSONResponse(content={"message": "Preferences saved successfully!"})

# ✅ Retrieve User Preferences
@app.get("/get-preference/")
async def get_preference(user_id: int, db: Session = Depends(get_db)):
    """Retrieve user UI preferences."""
    user = db.query(UserData).filter(UserData.id == user_id).first()
    if not user:
        return JSONResponse(content={"error": "User not found!"}, status_code=404)

    return JSONResponse(content={"user_id": user.id, "font_size": user.font_size, "high_contrast": user.high_contrast})


# ✅ Root Endpoint
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask/")
async def ask_question(
    request: Request,
    user_query: str = Form(...),
    literacy_level: str = Form(...),
    language: str = Form(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db)
):
    """ Save user question and generate AI response """

    # ✅ Debugging: Print received user_id
    print(f"📌 Received user_id in /ask/: {user_id}")

    # ✅ Check if `user_id` is valid
    if not user_id or user_id <= 0:
        return JSONResponse(content={"error": "Invalid or missing user_id"}, status_code=400)

    # ✅ Ensure user exists in DB
    user = db.query(UserData).filter(UserData.id == user_id).first()
    if not user:
        print(f"❌ User with ID {user_id} not found in /ask/")
        return JSONResponse(content={"error": "User not found!"}, status_code=404)

    # ✅ Store the user question
    new_query = UserQuery(user_id=user.id, user_query=user_query, language=language)
    db.add(new_query)
    db.commit()
    db.refresh(new_query)

    print(f"✅ Question Saved for User ID {user_id}: {user_query}")

    # ✅ Call Mistral API to Generate Response
    ai_response = generate_learning_content(user_query, literacy_level, language)

    return JSONResponse(content={
        "message": "Query saved successfully!",
        "user_id": user_id,
        "user_query": user_query,
        "ai_response": ai_response  # ✅ Include AI response
    })

from fastapi.responses import JSONResponse
from googletrans import Translator



@app.post("/translate/")
async def translate_response(ai_response: str = Form(...)):
    """ Translate AI response to Swahili """

    translator = Translator()

    try:
        #print(f"🔹 Translating Response: {ai_response}")

        # ✅ Keep `await` since it works for you
        translated = translator.translate(ai_response, src="en", dest="sw")
        translated_text = translated.text  # ✅ Extract translated text properly

        #print(f"✅ Translation Complete: {translated_text}")

        return JSONResponse(content={"translated_response": translated_text})  # ✅ Return JSON

    except Exception as e:
        print("❌ Translation Failed:", str(e))
        return JSONResponse(content={"translated_response": f"⚠️ Translation service is unavailable. Error: {str(e)}"})

import pyttsx3

def text_to_speech(text):
    """ Converts text to speech using pyttsx3 if offline, else gTTS """
    try:
        speech = gTTS(text=text, lang="en")
        speech.save("response.mp3")
        return "response.mp3"
    except Exception as e:
        print(f"❌ gTTS Failed, using Offline TTS: {e}")
        engine = pyttsx3.init()
        audio_path = "response.mp3"
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        return audio_path


from pydantic import BaseModel

# Define a Pydantic model to enforce correct request format

class SpeakRequest(BaseModel):
    text: str  # Expecting a string input

# ✅ Generate Speech Route
class SpeakRequest(BaseModel):
    text: str  # Expecting a string input

@app.post("/speak/")
async def generate_audio(request: SpeakRequest):
    """ Convert AI-generated text to speech """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="❌ Input text cannot be empty.")

        print(f"🗣 Generating speech for: {request.text}")  # ✅ Debugging: Print the text being used

        audio_path = "response.mp3"

        # ✅ Ensure the file doesn't already exist
        if os.path.exists(audio_path):
            os.remove(audio_path)  # Remove any existing file

        # ✅ Generate speech file
        try:
            speech = gTTS(text=request.text, lang="en")
            speech.save(audio_path)
        except Exception as gtts_error:
            print(f"❌ gTTS Error: {gtts_error}")  # ✅ Print gTTS issue
            raise HTTPException(status_code=500, detail=f"❌ Error generating speech: {gtts_error}")

        # ✅ Verify if the file was actually saved
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="❌ Speech file was not generated.")

        print(f"✅ Speech file generated successfully: {audio_path}")  # ✅ Confirm file saved

        return FileResponse(audio_path, media_type="audio/mpeg", filename="response.mp3")

    except Exception as e:
        print(f"❌ General Error: {e}")  # ✅ Print any other error
        raise HTTPException(status_code=500, detail=f"❌ Error generating speech: {str(e)}")

# ✅ Allow GET Requests (Optional)
from fastapi.responses import FileResponse
from gtts import gTTS
import os

@app.post("/speak/")
async def generate_audio(request: SpeakRequest):
    """ Convert AI-generated text to speech & return MP3 file """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="❌ Input text cannot be empty.")

        print(f"🗣 Generating speech for: {request.text}")  # ✅ Debugging: Print the text being used

        # Generate speech file
        audio_path = "response.mp3"
        speech = gTTS(text=request.text, lang="en")
        speech.save(audio_path)

        # Ensure file exists before returning
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="❌ Speech file was not generated.")

        return FileResponse(audio_path, media_type="audio/mpeg", filename="response.mp3")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error generating speech: {str(e)}")


# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

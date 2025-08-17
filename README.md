
# 🌍 Multi-Agent Travel Assistant

An Agentic AI Project built with **Streamlit, LangGraph, LangChain, and Groq**.
It uses multiple specialized AI agents that collaborate to provide **travel assistance** — including itinerary planning, budget suggestions, hotel/flight info, and local recommendations.

---

## 🚀 Features

* Multi-Agent System: different agents handle flights, hotels, budgets, itineraries, etc.
* Fast LLM Inference: powered by Groq API for speed.
* Interactive Frontend: Streamlit-based UI for user-friendly interaction.
* Secure API Key Handling: environment variables managed with `.env` (ignored in Git).

---

## 📂 Project Structure

* `app.py` → Main Streamlit app entry point
* `agents.py` → Multi-agent logic
* `requirements.txt` → Python dependencies
* `.env` → Private API keys (ignored in GitHub)
* `.env.example` → Example file with placeholders
* `.gitignore` → Ensures secrets and venv are ignored

---

## 🔧 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<ypur_user_name>/Multi-Agent-Travel-Assistant.git
   cd Multi-Agent-Travel-Assistant
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv .venv
   # Activate
   .venv\Scripts\activate      # Windows PowerShell
   source .venv/bin/activate   # Mac/Linux
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**
   Create a `.env` file in the root folder:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here   # optional
   ```

   (⚠️ Don’t upload `.env` to GitHub. Use `.env.example` for placeholders.)

5. **Run the App**

   ```bash
   streamlit run app.py
   ```

   Open [http://localhost:8501](http://localhost:8501)

---

## 📜 License

This project is for educational and experimental purposes. Add a license (e.g., MIT) if you plan to open-source it.

Do you also want me to make a **short version** (like 8–10 lines) that looks professional for GitHub, instead of this full guide?

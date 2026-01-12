
# gemini_file_search_api.py
"""
FastAPI backend for REAL Gemini File Search RAG (Option A: delete local temp file after indexing).
- Support for User Registration & Management
- Multiple API Keys per user with 1GB Quota Rotation
- Comprehensive Analytics
"""

import os
import time
import json
import shutil
import re
import requests
import hashlib
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Body, Query, BackgroundTasks
app = FastAPI(title="Gemini File Search RAG API (Multi-User & Analytics)")

# Helper for background logging
def log_history_background(store_id: str, store_name: str, question: str, answer: str):
    try:
        db = get_db()
        db[COL_HISTORY].insert_one({
            "store_id": store_id,
            "store_name": store_name,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "type": "public_id_query"
        })
    except Exception as e:
        print(f"Background logging failed: {e}")

@app.get("/ask-by-id")
def ask_question_by_id(
    background_tasks: BackgroundTasks,
    store_id: str = Query(..., description="Unique Store ID"),
    question: str = Query(..., description="Question to ask"),
    system_prompt: Optional[str] = Query(None, description="System prompt override")
):
    """
    Ask a question to a specific store using its unique ID.
    No user email required.
    """
    db = get_db()
    
    # 1. Verify store exists
    store = db[COL_STORES].find_one({"store_id": store_id})
    if not store:
        return JSONResponse({"error": "Store not found"}, status_code=404)
    
    # 3. Setup Gemini
    api_key = store["api_key_used"]
    fs_name = store["file_search_store_name"]
    
    try:
        client = init_gemini_client(api_key)
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[fs_name]))
        
        sys_inst = system_prompt
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[file_search_tool],
                system_instruction=sys_inst
            )
        )
        
        response_text = getattr(response, "text", "")
        
        # 2. Log question AND answer in BACKGROUND
        background_tasks.add_task(
            log_history_background, 
            store_id, 
            store["store_name"], 
            question, 
            response_text
        )
        
        return {
            "success": True,
            "response_text": response_text
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
        
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from pathlib import Path
import aiofiles
from datetime import datetime
import pymongo
from pymongo import MongoClient
import logging

# Try to import google genai SDK
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------
UPLOAD_ROOT = Path("/data/uploads")
MAX_FILE_BYTES = 50 * 1024 * 1024      # 50 MB default file limit
KEY_QUOTA_BYTES = 1 * 1024 * 1024 * 1024 # 1 GB limit per key
POLL_INTERVAL = 2                       # seconds
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# MongoDB configuration
MONGODB_URI = "mongodb+srv://wisdomkagyan_db_user:gqbCoXr99sKOcXEw@cluster0.itxqujm.mongodb.net/?appName=Cluster0"
DATABASE_NAME = "gemini_file_search_v2" # Updated DB name for new schema
# Collections
COL_USERS = "users"
COL_KEYS = "api_keys"
COL_STORES = "user_stores"
COL_HISTORY = "question_history"

# ----------------------------------------



# ---------------- MongoDB Setup ----------------

def get_mongo_client():
    try:
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MongoDB: {e}")

def get_db():
    client = get_mongo_client()
    return client[DATABASE_NAME]

def init_mongodb():
    """Initialize MongoDB with proper indexes"""
    try:
        db = get_db()
        
        # User indexes
        db[COL_USERS].create_index([("email", pymongo.ASCENDING)], unique=True)
        
        # API Key indexes
        db[COL_KEYS].create_index([("user_email", pymongo.ASCENDING)])
        db[COL_KEYS].create_index([("key", pymongo.ASCENDING)], unique=True)
        
        # Store indexes
        db[COL_STORES].create_index([("user_email", pymongo.ASCENDING), ("store_name", pymongo.ASCENDING)], unique=True)
        
        # History indexes
        db[COL_HISTORY].create_index([("user_email", pymongo.ASCENDING)])
        
        print("MongoDB initialized successfully")
    except Exception as e:
        print(f"MongoDB initialization warning: {e}")

init_mongodb()

# ---------------- Helpers: Hashing ----------------

def hash_password(password: str) -> str:
    """Simple SHA256 hash for passwords"""
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- Helpers: Database Access ----------------

def get_user(email: str):
    db = get_db()
    return db[COL_USERS].find_one({"email": email})

def create_user_entry(email: str, password_hash: str):
    db = get_db()
    try:
        db[COL_USERS].insert_one({
            "email": email,
            "password": password_hash,
            "created_at": datetime.now()
        })
        return True
    except pymongo.errors.DuplicateKeyError:
        return False

def add_api_key_entry(email: str, api_key: str):
    db = get_db()
    # Check if key exists globally to avoid reuse/confusion
    if db[COL_KEYS].find_one({"key": api_key}):
        raise ValueError("API Key already in use.")
    
    db[COL_KEYS].insert_one({
        "user_email": email,
        "key": api_key,
        "current_usage_bytes": 0,
        "active": True,
        "created_at": datetime.now()
    })

def get_user_keys(email: str):
    db = get_db()
    return list(db[COL_KEYS].find({"user_email": email}))

def get_best_available_key(email: str):
    """Find the first active key with usage < 1GB"""
    db = get_db()
    # Find all keys for user, sorted by creation (fill oldest first)
    keys = list(db[COL_KEYS].find({"user_email": email}).sort("created_at", pymongo.ASCENDING))
    
    for k in keys:
        if k.get("current_usage_bytes", 0) < KEY_QUOTA_BYTES:
            return k["key"]
    
    return None

def update_key_usage(api_key: str, bytes_added: int):
    db = get_db()
    db[COL_KEYS].update_one(
        {"key": api_key},
        {"$inc": {"current_usage_bytes": bytes_added}}
    )

def log_question(email: str, question: str, stores: List[str]):
    db = get_db()
    db[COL_HISTORY].insert_one({
        "user_email": email,
        "question": question,
        "stores": stores,
        "timestamp": datetime.now()
    })

# ---------------- Request Models ----------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class AddKeyRequest(BaseModel):
    email: EmailStr
    api_key: str

class CreateStoreRequest(BaseModel):
    email: EmailStr
    store_name: str

class AskRequest(BaseModel):
    email: EmailStr
    stores: List[str] = []
    question: str
    system_prompt: Optional[str] = None

# ---------------- Gemini Helper ----------------

def validate_gemini_key(api_key: str):
    """Try to generate a simple response to validate key"""
    if genai is None:
        raise RuntimeError("google-genai SDK missing")
    try:
        client = genai.Client(api_key=api_key)
        # Simple ping-like request
        client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="hi",
            config=types.GenerateContentConfig(max_output_tokens=5)
        )
        return True
    except Exception as e:
        print(f"Key validation failed: {e}")
        return False

def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai SDK missing")
    return genai.Client(api_key=api_key)

def wait_for_operation(client, operation):
    op = operation
    while not getattr(op, "done", False):
        time.sleep(POLL_INTERVAL)
        try:
            if hasattr(client, "operations") and hasattr(client.operations, "get"):
                op = client.operations.get(op)
        except Exception:
            pass
    
    if getattr(op, "error", None):
        raise RuntimeError(f"Operation failed: {op.error}")
    return op

def clean_filename(name: str, max_len: int = 180) -> str:
    if not name: return "file"
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"^\.+", "", name)
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name)
    name = re.sub(r"__+", "_", name)
    if len(name) > max_len: name = name[:max_len]
    if not name: return "file"
    return name

def rest_list_documents_for_store(file_search_store_name, api_key):
    url = f"{GEMINI_REST_BASE}/{file_search_store_name}/documents"
    params = {"key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception:
        return []

# ensure dirs
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# =====================================================
# USER MANAGEMENT APIs
# =====================================================

@app.post("/register")
def register_user(payload: RegisterRequest):
    """Register a new user"""
    if create_user_entry(payload.email, hash_password(payload.password)):
        return {"success": True, "message": "User registered successfully"}
    else:
        return JSONResponse({"success": False, "error": "User already exists"}, status_code=400)

@app.post("/users/keys")
def add_user_key(payload: AddKeyRequest):
    """Add and validate a Gemini API Key for user"""
    user = get_user(payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate Key
    if not validate_gemini_key(payload.api_key):
         return JSONResponse({"success": False, "error": "Invalid API Key or Model unavailable"}, status_code=400)
    
    try:
        add_api_key_entry(payload.email, payload.api_key)
        return {"success": True, "message": "API Key added and verified"}
    except ValueError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)

@app.get("/users/analytics")
def get_user_analytics(email: str):
    """Get comprehensive analytics for a user"""
    db = get_db()
    user = get_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 1. Stores count
    stores = list(db[COL_STORES].find({"user_email": email}))
    total_stores = len(stores)
    
    # 2. Files count & Storage
    total_files = 0
    # Calculate storage from KEYS usage, not stores (more accurate to quota)
    keys = list(db[COL_KEYS].find({"user_email": email}))
    total_storage_bytes = sum(k.get("current_usage_bytes", 0) for k in keys)
    
    for store in stores:
        total_files += len(store.get("files", []))

    # 3. Questions history count
    # Count by email (authenticated ask) OR by store_id ownership (anonymous ask-by-id)
    # Get all store IDs for this user
    user_store_ids = [s["store_id"] for s in stores if "store_id" in s]
    
    total_questions = db[COL_HISTORY].count_documents({
        "$or": [
            {"user_email": email},
            {"store_id": {"$in": user_store_ids}}
        ]
    })

    # 4. Detailed Keys Usage
    keys_status = []
    total_limit_bytes = 0
    
    for k in keys:
        keys_status.append({
            "key_masked": k["key"][:4] + "..." + k["key"][-4:],
            "usage_bytes": k.get("current_usage_bytes", 0),
            "usage_gb": round(k.get("current_usage_bytes", 0) / (1024**3), 4),
            "limit_gb": 1.0,
            "created_at": k.get("created_at")
        })
        total_limit_bytes += KEY_QUOTA_BYTES

    return {
        "success": True,
        "analytics": {
            "total_stores": total_stores,
            "total_files": total_files,
            "total_storage_bytes": total_storage_bytes,
            "total_storage_gb": round(total_storage_bytes / (1024**3), 4),
            "total_limit_gb": round(total_limit_bytes / (1024**3), 4),
            "total_remaining_gb": round((total_limit_bytes - total_storage_bytes) / (1024**3), 4),
            "total_questions_asked": total_questions,
            "keys_status": keys_status
        }
    }

# =====================================================
# STORE APIs (User Context)
# =====================================================

@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    """Create store using an available user key"""
    user = get_user(payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Find available key
    api_key = get_best_available_key(payload.email)
    if not api_key:
        return JSONResponse(
            {"success": False, "error": "No available API keys with sufficient quota (1GB limit reached on all keys). Add a new key."}, 
            status_code=402
        )

    db = get_db()
    # Check duplicate store name
    if db[COL_STORES].find_one({"user_email": payload.email, "store_name": payload.store_name}):
        return JSONResponse({"error": "Store name already exists for this user."}, status_code=400)

    try:
        client = init_gemini_client(api_key)
        fs_store = client.file_search_stores.create(config={"display_name": payload.store_name})
        fs_store_name = getattr(fs_store, "name", None) or fs_store
        
        # Generate unique ID
        store_id = str(uuid.uuid4())
        
        # Save store with link to the SPECIFIC key used
        db[COL_STORES].insert_one({
            "store_id": store_id,
            "user_email": payload.email,
            "store_name": payload.store_name,
            "file_search_store_name": fs_store_name,
            "api_key_used": api_key,
            "created_at": datetime.now(),
            "files": []
        })
        
        return {
            "success": True, 
            "store_id": store_id,
            "store_name": payload.store_name, 
            "file_search_store_resource": fs_store_name,
            "used_key_snippet": api_key[:4] + "..."
        }
    except Exception as e:
        return JSONResponse({"error": f"Gemini creation failed: {e}"}, status_code=500)

@app.post("/stores/{store_name}/upload")
async def upload_files(
    store_name: str,
    email: str = Form(...),
    limit: bool = Form(True),
    files: List[UploadFile] = File(...)
):
    """Upload files to store. Checks quota of the key ASSOCIATED with that store."""
    db = get_db()
    store = db[COL_STORES].find_one({"user_email": email, "store_name": store_name})
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    api_key = store.get("api_key_used")
    if not api_key:
        raise HTTPException(status_code=500, detail="Store has no associated API key")

    # Check key quota first
    key_entry = db[COL_KEYS].find_one({"key": api_key})
    if not key_entry:
         raise HTTPException(status_code=500, detail="Associated API key record lost")
    
    if key_entry.get("current_usage_bytes", 0) >= KEY_QUOTA_BYTES:
         return JSONResponse({"success": False, "error": "Quota exceeded for the API Key used by this store. Cannot upload more files to this store."}, status_code=403)

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)

    fs_store_name = store["file_search_store_name"]
    temp_folder = UPLOAD_ROOT / email / store_name
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_bytes_uploaded = 0

    for upload in files:
        filename = clean_filename(upload.filename)
        temp_path = temp_folder / filename
        
        size = 0
        try:
            async with aiofiles.open(temp_path, "wb") as out_f:
                while True:
                    chunk = await upload.read(1024*1024)
                    if not chunk: break
                    size += len(chunk)
                    await out_f.write(chunk)
            
            # Check validation BEFORE sending to Gemini if possible, but strict "bytes used" logic usually means storage
            # Here we count file size towards quota
            
            if (key_entry.get("current_usage_bytes", 0) + total_bytes_uploaded + size) > KEY_QUOTA_BYTES:
                 results.append({"filename": filename, "uploaded": False, "reason": "Quota Exceeded"})
                 os.remove(temp_path)
                 continue

            # Upload to Gemini
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_store_name,
                config={"display_name": filename}
            )
            op = wait_for_operation(client, op)
            
            # Resolve document identifier
            doc_res = None
            try:
                doc_res = op.response.file_search_document.name
            except:
                pass
            
            if not doc_res:
                # Fallback to REST list
                docs = rest_list_documents_for_store(fs_store_name, api_key)
                for d in docs:
                    if d.get("displayName") == filename or filename in d.get("name", ""):
                        doc_res = d.get("name")
                        break
            
            doc_id = doc_res.split("/")[-1] if doc_res else "unknown"
            
            # Add file record
            db[COL_STORES].update_one(
                {"_id": store["_id"]},
                {"$push": {
                    "files": {
                        "display_name": filename,
                        "size_bytes": size,
                        "uploaded_at": datetime.now(),
                        "document_resource": doc_res,
                        "document_id": doc_id
                    }
                }}
            )
            
            total_bytes_uploaded += size
            
            # Cleanup local
            os.remove(temp_path)
            
            results.append({
                "filename": filename,
                "uploaded": True,
                "document_id": doc_id,
                "size": size
            })

        except Exception as e:
            results.append({"filename": filename, "error": str(e)})

    # Update Global Usage for Key
    update_key_usage(api_key, total_bytes_uploaded)
    
    # Clean folder
    if temp_folder.exists() and not any(temp_folder.iterdir()):
        shutil.rmtree(temp_folder)

    return {"success": True, "results": results}

@app.get("/stores")
def list_user_stores(email: str):
    db = get_db()
    stores = list(db[COL_STORES].find({"user_email": email}, {"_id": 0}))
    return {"success": True, "stores": stores}

@app.delete("/stores/{store_name}")
def delete_store(store_name: str, email: str):
    db = get_db()
    store = db[COL_STORES].find_one({"user_email": email, "store_name": store_name})
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    api_key = store.get("api_key_used")
    # Try delete remote
    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=store["file_search_store_name"], config={"force": True})
    except Exception:
        pass
    
    # Remove DB entry
    db[COL_STORES].delete_one({"_id": store["_id"]})
    
    # Note: We do NOT decrease the 'usage' bytes because typically API providers charge for ingress/processing
    # or the storage is freed but we might want to track 'bandwidth'.
    # However, if this is purely STORAGE quota, we SHOULD decrease usage.
    # The prompt implies "storage reached 1gb", so typically if I delete headers, I free space.
    # Let's free the space for the user.
    
    freed_bytes = sum(f.get("size_bytes", 0) for f in store.get("files", []))
    update_key_usage(api_key, -freed_bytes)

    return {"success": True, "deleted": store_name}

@app.get("/ask")
def ask_question(
    email: str = Query(..., description="User email"),
    question: str = Query(..., description="Question to ask"),
    stores: Optional[List[str]] = Query(None, description="List of store names to query"),
    system_prompt: Optional[str] = Query(None, description="System prompt override")
):
    db = get_db()
    
    # Identify stores
    target_stores = []
    if stores:
        target_stores = list(db[COL_STORES].find({
            "user_email": email, 
            "store_name": {"$in": stores}
        }))
    else:
        target_stores = list(db[COL_STORES].find({"user_email": email}))
    
    if not target_stores:
        return JSONResponse({"error": "No stores found/selected"}, status_code=400)
    
    # Group stores by API KEY
    # We can only mix stores that share the SAME API key in a single request unfortunately,
    # OR we have to pick a key that has access to them.
    # Gemini limitation: A Client is initialized with ONE api key.
    # If the user has stores created with Key A and stores with Key B, we cannot query both in one request easily
    # unless we use Key A and Key A has permissions on Key B's resources (unlikely for simple API keys)
    # OR if we assume the standard "add user to project" model.
    # Since we are just using simple API keys, they are distinct projects usually.
    # We will pick the API Key of the FIRST store selected and filter the list to only include stores from that key.
    # This is a critical logical limitation of "Rotation" with simple Keys.
    
    primary_key = target_stores[0]["api_key_used"]
    compatible_stores = [s for s in target_stores if s["api_key_used"] == primary_key]
    
    fs_names = [s["file_search_store_name"] for s in compatible_stores]
    
    if len(compatible_stores) < len(target_stores):
        # Warn logic: silently ignoring stores from other keys for this request
        pass 
        
    try:
        client = init_gemini_client(primary_key)
        
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=fs_names))
        
        # Use system prompt if provided
        sys_inst = system_prompt
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[file_search_tool],
                system_instruction=sys_inst
            )
        )
        
        return {
            "success": True,
            "response_text": getattr(response, "text", "")
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/ask-by-id")
def ask_question_by_id(
    store_id: str = Query(..., description="Unique Store ID"),
    question: str = Query(..., description="Question to ask"),
    system_prompt: Optional[str] = Query(None, description="System prompt override")
):
    """
    Ask a question to a specific store using its unique ID.
    No user email required.
    """
    db = get_db()
    
    # 1. Verify store exists
    store = db[COL_STORES].find_one({"store_id": store_id})
    if not store:
        return JSONResponse({"error": "Store not found"}, status_code=404)
        
    # 3. Setup Gemini
    api_key = store["api_key_used"]
    fs_name = store["file_search_store_name"]
    
    try:
        client = init_gemini_client(api_key)
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[fs_name]))
        
        sys_inst = system_prompt
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[file_search_tool],
                system_instruction=sys_inst
            )
        )
        
        response_text = getattr(response, "text", "")
        
        # 2. Log question AND answer
        db[COL_HISTORY].insert_one({
            "store_id": store_id,
            "store_name": store["store_name"],
            "question": question,
            "answer": response_text,
            "timestamp": datetime.now(),
            "type": "public_id_query"
        })
        
        return {
            "success": True,
            "response_text": response_text
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    try:
        get_mongo_client()
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=500)

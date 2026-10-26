import os
import json
from pathlib import Path
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_file(local_path, storage_path, content_type):
    with open(local_path, "rb") as f:
        supabase.storage.from_("scans").upload(
            storage_path,
            f,
            {"content-type": content_type, "upsert": "true"},
        )

    return supabase.storage.from_("scans").get_public_url(storage_path)


def save_scan_to_supabase(user_id, scan_name, image_path, json_path):
    image_storage_path = f"{user_id}/{scan_name}/{scan_name}.png"
    json_storage_path = f"{user_id}/{scan_name}/{scan_name}_detcls.json"

    image_url = upload_file(image_path, image_storage_path, "image/png")
    upload_file(json_path, json_storage_path, "application/json")

    supabase.table("scans").insert({
        "user_id": user_id,
        "scan_name": scan_name,
        "image_path": image_storage_path,
        "json_path": json_storage_path,
    }).execute()

    return image_url


def get_user_scans(user_id):
    response = (
        supabase.table("scans")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data


def get_public_image_url(image_path):
    return supabase.storage.from_("scans").get_public_url(image_path)


def load_scan_results(json_path):
    data = supabase.storage.from_("scans").download(json_path)
    return json.loads(data.decode("utf-8"))


def delete_scan(scan):
    supabase.storage.from_("scans").remove([
        scan["image_path"],
        scan["json_path"],
    ])

    supabase.table("scans").delete().eq("id", scan["id"]).execute()
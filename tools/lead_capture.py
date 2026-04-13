import csv
import os
from datetime import datetime

LEADS_FILE = "leads.csv"

def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """Simulate saving a captured lead. Called only after all 3 fields are collected."""
    
    # Print to terminal
    print(f"\n✅ Lead captured successfully!")
    print(f"   Name: {name}")
    print(f"   Email: {email}")
    print(f"   Platform: {platform}\n")

    # Save to CSV
    file_exists = os.path.exists(LEADS_FILE)

    with open(LEADS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Email", "Platform", "Timestamp"])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Name": name,
            "Email": email,
            "Platform": platform,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    print(f"📁 Lead saved to leads.csv\n")
    

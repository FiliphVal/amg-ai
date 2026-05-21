import json
import os
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi

def download_all_transcripts():
    output_dir = "transcripts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        with open("amg_video_ids.json", "r", encoding="utf-8") as f:
            all_video_ids = json.load(f)
    except FileNotFoundError:
        print("Kunde inte hitta listan. Kör get_video_ids.py först!")
        return

    video_ids = all_video_ids[:420]
    print("Startar nedladdning (Manuell VPN-strategi)...\n")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    ytt_api = YouTubeTranscriptApi()

    # Vi går tillbaka till en hederlig for-loop
    for index, v_id in enumerate(video_ids, start=1):
        filepath = os.path.join(output_dir, f"{v_id}.txt")
        
        # Om filen finns sedan tidigare VPN-körningar, hoppa över!
        if os.path.exists(filepath):
            print(f"[{index}/{len(video_ids)}] {v_id} finns redan.")
            skipped_count += 1
            continue
            
        print(f"[{index}/{len(video_ids)}] Hämtar: {v_id}...", end=" ", flush=True)
        
        try:
            transcript = ytt_api.fetch(v_id)
            
            try:
                full_text = " ".join([line.text for line in transcript])
            except AttributeError:
                full_text = " ".join([line['text'] for line in transcript])
            
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(full_text)
                
            print("OK!")
            success_count += 1
            
        except Exception as e:
            # Här ser du direkt när din nuvarande VPN-adress blir blockerad
            print("MISSLYCKADES!")
            print(f"   -> Orsak: {e}")
            fail_count += 1
            
        # Standardpaus
        time.sleep(random.uniform(2.0, 4.0))

    print("\n" + "="*40)
    print("KÖRNING KLAR ELLER AVBRUTEN!")
    print(f"Nyhämtade: {success_count}")
    print(f"Redan existerande: {skipped_count}")
    print(f"Misslyckade: {fail_count}")
    print("="*40)

if __name__ == "__main__":
    download_all_transcripts()
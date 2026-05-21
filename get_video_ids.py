import subprocess
import json

channel_url = "https://www.youtube.com/@AthleticMotionGolf/videos"

def get_recent_videos():
    print("Hämtar de 500 senaste riktiga videorna (filtrerar bort Shorts)...")
    
    # --match-filter "duration > 60": Sorterar bort alla videor som är 60 sekunder eller kortare
    cmd = [
        "yt-dlp", 
        "--flat-playlist", 
        "--match-filter", "duration > 60", 
        "--playlist-end", "500", 
        "--print", "id", 
        channel_url
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Ett fel uppstod i yt-dlp: {result.stderr}")
        return

    video_ids = [vid.strip() for vid in result.stdout.strip().split('\n') if vid.strip()]

    print(f"\nHittade totalt {len(video_ids)} långa videor!")
    
    with open("amg_video_ids.json", "w", encoding="utf-8") as f:
        json.dump(video_ids, f, indent=4)
        
    print("Listan med IDn är sparad i 'amg_video_ids.json'")

if __name__ == "__main__":
    get_recent_videos()
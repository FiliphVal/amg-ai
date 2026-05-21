from youtube_transcript_api import YouTubeTranscriptApi

video_id = "q_4YJ-AXSPA"

def fetch_single_transcript(v_id):
    print(f"Hämtar transkribering för video: {v_id}...")
    
    try:
        transcript = YouTubeTranscriptApi().fetch(v_id)
        
        # Här är den avgörande fixen: line.text istället för line['text']
        full_text = " ".join([line.text for line in transcript])
        
        filename = f"transcript_{v_id}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(full_text)
            
        print(f"\nSUCCÉ! Texten har sparats i filen: {filename}")
        print("*" * 40)
        print("Smakprov på texten:")
        print(full_text[:200] + "...")
        print("*" * 40)
        
    except Exception as e:
        print(f"\nNågot gick fel vid hämtningen: {e}")

if __name__ == "__main__":
    fetch_single_transcript(video_id)
import pyttsx3


class VoiceEngine:
    def __init__(self) -> None:                
        self.engine = pyttsx3.init('sapi5')
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)
        self.engine.setProperty("rate", 220)


    def speak(self, audio:str):
        self.engine.say(audio)
        try:
            self.engine.runAndWait()
        except RuntimeError:
            pass

        
        
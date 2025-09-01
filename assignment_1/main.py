import cv2, os

class Oppgave:
    def __init__(self, bildefil):
        self.bildefil = bildefil

    def skriv_bilde_info(self):
        bilde = cv2.imread(self.bildefil)
        if bilde is None:
            print("Feil: fant ikke bilde!")
            return
        h, b, k = bilde.shape
        print(f"Høyde: {h}\nBredde: {b}\nKanaler: {k}\nStørrelse: {bilde.size}\nDatatype: {bilde.dtype}")

    def lagre_kamera_info(self):
        cam = cv2.VideoCapture(0)
        fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30
        b, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs("solutions", exist_ok=True)
        with open("solutions/camera_outputs.txt", "w") as f:
            f.write(f"fps: {fps}\nHøyde: {h}\nBredde: {b}\n")
        print("Kamera-info lagret ✔")
        cam.release()

def main():
    oppg = Oppgave("solutions/lena-1.png")
    oppg.skriv_bilde_info()
    oppg.lagre_kamera_info()

if __name__ == "__main__":
    main()
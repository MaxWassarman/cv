#Following openbiomechanics computer vision

#pip install face_recognition
#Go through info in face_recognition github to install
from PIL import Image
import face_recognition

image = face_recognition.load_image_file("manpic.jpeg")
face_locations = face_recognition.face_locations(image)

print("I found {} face(s)".format(len(face_locations)))

for i in face_locations:
    top, right, bottom, left = i

face_image = image[top:bottom, left:right]
pil_image = Image.fromarray(face_image)
pil_image.show()
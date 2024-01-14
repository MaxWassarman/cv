#Following openbiomechanics computer vision

#pip install face_recognition
#Go through info in face_recognition github to install
from PIL import Image, ImageDraw, ImageFont
import face_recognition

image = face_recognition.load_image_file("manpic.jpeg")
face_locations = face_recognition.face_locations(image)

print("I found {} face(s)".format(len(face_locations)))

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

text_height = 5

for face_location in face_locations:
    top, right, bottom, left = face_location
   
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))

    text = "Person"
    text_width = len(text) * 7

    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
    draw.text((left + 6, bottom - text_height - 5), text, fill=(255, 255, 255, 255))

pil_image.show()
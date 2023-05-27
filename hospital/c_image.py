import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
from django.core.files.storage import default_storage
from django.conf import settings
from . import forms,models


def crop_image(image_path, output_path):
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Select Crop Area")
    rs = RectangleSelector(ax, lambda eclick, erelease: onselect(eclick, erelease, image, ax, output_path), drawtype="box", useblit=True,
                           minspanx=5, minspany=5, spancoords="pixels", interactive=True)
    
    save_button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    save_button = plt.Button(save_button_ax, 'Save')
    save_button.on_clicked(lambda event: save_image(ax, rs, output_path))
    
    recrop_button_ax = plt.axes([0.6, 0.05, 0.2, 0.075])
    recrop_button = plt.Button(recrop_button_ax, 'Recrop')
    recrop_button.on_clicked(lambda event: recrop_image(image, ax, rs))
    
    plt.show()

def onselect(eclick, erelease, image, ax, output_path):
    
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    cropped_image = image.crop((x1, y1, x2, y2))
    ax.imshow(cropped_image)
    ax.set_title("Cropped Image")
    plt.draw()
    ax.cropped_image = cropped_image  # Store the cropped image in the axes object

def save_image(ax, rs, output_path):
    cropped_image = ax.cropped_image
    cropped_image.save(output_path)
    print(f"Image cropped and saved as '{output_path}'")

    #save cropped image's path into db
    patient_P = models.PatientPrescriptionData.objects.all().last()
    patient_P.cropped_image = output_path
    patient_P.save(update_fields=['cropped_image'])
    rs.disconnect_events()  # Disconnect the RectangleSelector events to terminate the program
    plt.close()

def recrop_image(image, ax, rs):
    ax.clear()
    ax.imshow(image)
    ax.set_title("Select Crop Area")
    rs.connect_event('button_press_event', rs.onpress)
    plt.draw()
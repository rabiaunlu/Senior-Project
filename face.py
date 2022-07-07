# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np


def scan_known_people_for_known_names(known_people_folder):

    known_names = []

    known_face_encodings = []

    encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        if os.path.exists("./encodings/" + basename + ".arr.npy"):
            encodings = np.array([np.load("./encodings/" + basename + ".arr.npy")])
        else:

            encodings = face_recognition.face_encodings(img, None, 4)
            np.save("./encodings/" + basename + ".arr", encodings[0])
        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
    return known_names


def scan_known_people_for_known_face_encoding(known_people_folder = "students/"):

    known_names = []

    known_face_encodings = []

    encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        if os.path.exists("./encodings/" + basename + ".arr.npy"):
            encodings = np.array([np.load("./encodings/" + basename + ".arr.npy")])
        else:
            encodings = face_recognition.face_encodings(img, None, 4)
            np.save("./encodings/" + basename + ".arr", encodings[0])
        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_face_encodings.append(encodings[0])
    return  known_face_encodings

def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):

    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)
    face_names = []
    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        match = face_recognition.compare_faces(known_face_encodings,
                                               unknown_encoding, 0.5)
        name = "Unknown"
        for k in range(len(match)):
            if match[k]:
              name = known_names[k]
        face_names.append(name)
    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, "no_persons_found", None, show_distance)

    return face_names



def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def scanClass(known_people_folder,image_to_check,class_crn):
    known_face_encodings = scan_known_people_for_known_face_encoding(known_people_folder)
    known_names = scan_known_people_for_known_names(known_people_folder)
    print("registered students : ")
    print (known_names)
    
    imagefile = open("/home/rabia/face_recognition/face_recognition/class/"+ class_crn + ".jpg","rb")
    image_to_check = imagefile
    facenames = test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False)
    print("students in class : ")
    print(facenames)

def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )
    pool.starmap(test_image, function_parameters)


@click.command()
#@click.argument('known_people_folder')
#@click.argument('image_to_check')
@click.argument('class_crn')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
@click.option('--image_to_check',default = "class/")
@click.option('--known_people_folder',default="students/")

def main(known_people_folder, image_to_check, cpus, tolerance, show_distance,class_crn):
    known_names = scan_known_people_for_known_names(known_people_folder)
    known_face_encodings = scan_known_people_for_known_face_encoding(known_people_folder)
    attendance = scanClass(known_people_folder,image_to_check,class_crn)
    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)


if __name__ == "__main__":
 main()
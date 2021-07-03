import cv2
import pathlib
from headpose import est_face_pose, detect_face_marks


if __name__ == "__main__" :
    def cv2_scatter(img, marks):
        for mark in marks:
            # pt = (int(mark[1]), int(mark[0]))
            pt = (int(mark[0]), int(mark[1]))
            cv2.circle(img, pt, 1, (0, 0, 255),thickness=10)
        # img = cv2.resize(img, (1000, 1000))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # est = PoseEstimator()  #load the model
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images/')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("18.jpg")))
    for image_path in TEST_IMAGE_PATHS:
        img = cv2.imread(str(image_path))
        # roll, pitch, yawn = est.pose_from_image(img)  # estimate the head pose
        marks = detect_face_marks(img)  # plot the image with the face detection marks
        if marks is not None :
            cv2_scatter(img, marks)
            img_name = str(image_path).split('/')[-1]
            cv2.imwrite('images/head_pose_est/'+img_name, img)
        else :
            print('no marks')
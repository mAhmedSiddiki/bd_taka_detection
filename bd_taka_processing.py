from PIL import Image
from cv2 import cv2
import numpy as np

def taka_1(img_path):
    return "Not Complete"


def taka_2(img_path):
    return "Not Complete"


def taka_5(img_path):
    return "Not Complete"


def taka_10(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 11
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)

    # Crop - 2
    left = right
    top = 0
    right = width / 3
    bottom = height / 1.25

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)

    # Crop - 3
    left = right
    top = 0
    right = width / 1.35
    bottom = height / 4.2

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.3
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)

    # Crop - 5
    left = right
    top = height / 1.35
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)

    # Crop - 6
    left = width / 1.1
    top = height / 4
    right = width
    bottom = height / 1.35

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)

    # Crop - 7
    left = width / 1.3
    top = 0
    right = width
    bottom = height / 4.5

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (92, 500))
            train_image = cv2.resize(train_image, (92, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (308, 500))
            test_image = cv2.resize(test_image, (308, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 144))
            train_image = cv2.resize(train_image, (500, 144))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 130))
            train_image = cv2.resize(train_image, (500, 130))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 277))
            train_image = cv2.resize(train_image, (500, 277))
        elif position == 6:
            test_image = cv2.resize(test_image, (188, 500))
            train_image = cv2.resize(train_image, (188, 500))
        else:
            test_image = cv2.resize(test_image, (500, 237))
            train_image = cv2.resize(train_image, (500, 237))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 5):
            matching_image(cv2.imread("static/Cutting/taka_10/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
        # print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))
    

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))
    

    # print()

    
    original_500_taka_list = [12.00, 18.00, 30.00, 1.00, 12.00, 2.00, 10.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_500_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def taka_20(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 9
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)

    # Crop - 2
    left = right
    top = 0
    right = width / 2.7
    bottom = height / 1.25

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)

    # Crop - 3
    left = right
    top = 0
    right = width / 1.35
    bottom = height / 4.5

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.3
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)

    # Crop - 5
    left = right
    top = height / 1.35
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)

    # Crop - 6
    left = width / 1.1
    top = height / 4
    right = width
    bottom = height / 1.35

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)

    # Crop - 7
    left = width / 1.3
    top = 0
    right = width
    bottom = height / 4.5

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (122, 500))
            train_image = cv2.resize(train_image, (122, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (355, 500))
            test_image = cv2.resize(test_image, (355, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 137))
            train_image = cv2.resize(train_image, (500, 137))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 132))
            train_image = cv2.resize(train_image, (500, 132))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 283))
            train_image = cv2.resize(train_image, (500, 283))
        elif position == 6:
            test_image = cv2.resize(test_image, (202, 500))
            train_image = cv2.resize(train_image, (202, 500))
        else:
            test_image = cv2.resize(test_image, (500, 220))
            train_image = cv2.resize(train_image, (500, 220))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111", "Crop_111111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222", "Crop_222222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333", "Crop_333333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444", "Crop_444444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555", "Crop_555555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666", "Crop_666666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777", "Crop_777777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 6):
            matching_image(cv2.imread("static/Cutting/taka_20/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)
            

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
        # print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))
    

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))
    

    # print()

    # Checking note original or fake
    original_500_taka_list = [20.00, 11.00, 37.00, 9.00, 10.00, 5.00, 12.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_500_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def taka_50(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 10.5
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)
    # img_crop_1.save("Crop/Crop_1.jpg")

    # Crop - 2
    left = right
    top = 0
    right = width / 3.1
    bottom = height / 1.2

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)
    # img_crop_2.save("Crop/Crop_2.jpg")

    # Crop - 3
    left = right
    top = 0
    right = width / 1.3
    bottom = height / 4

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)
    # img_crop_3.save("Crop/Crop_3.jpg")

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.28
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)
    # img_crop_4.save("Crop/Crop_4.jpg")

    # Crop - 5
    left = right
    top = height / 1.35
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)
    # img_crop_5.save("Crop/Crop_5.jpg")

    # Crop - 6
    left = width / 1.15
    top = height / 3.4
    right = width
    bottom = height / 1.3

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)
    # img_crop_6.save("Crop/Crop_6.jpg")

    # Crop - 7
    left = width / 1.3
    top = 0
    right = width
    bottom = height / 4.5

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)
    # img_crop_7.save("Crop/Crop_7.jpg")

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (103, 500))
            train_image = cv2.resize(train_image, (103, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (295, 500))
            test_image = cv2.resize(test_image, (295, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 129))
            train_image = cv2.resize(train_image, (500, 129))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 116))
            train_image = cv2.resize(train_image, (500, 116))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 274))
            train_image = cv2.resize(train_image, (500, 274))
        elif position == 6:
            test_image = cv2.resize(test_image, (298, 500))
            train_image = cv2.resize(train_image, (298, 500))
        else:
            test_image = cv2.resize(test_image, (500, 222))
            train_image = cv2.resize(train_image, (500, 222))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111", "Crop_111111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222", "Crop_222222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333", "Crop_333333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444", "Crop_444444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555", "Crop_555555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666", "Crop_666666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777", "Crop_777777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 6):
            matching_image(cv2.imread("static/Cutting/taka_50/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)
            # cv2.imshow("test_image - " + str(step) + " - " + str(j), tes_img[step][j])
            # cv2.imshow("train_image - " + str(step) + " - " + str(j), tr_img[step][j])
            # cv2.imshow("result - " + str(step) + " - " + str(j), res_img[step][j])

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
    #     print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))

    # print()

    # Checking note original or fake
    original_500_taka_list = [23.00, 20.00, 34.00, 12.00, 12.00, 7.00, 17.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_500_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def taka_100(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 5.5
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)

    # Crop - 2
    left = right
    top = 0
    right = width / 2.6
    bottom = height / 1.2

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)

    # Crop - 3
    left = right
    top = 0
    right = width / 1.35
    bottom = height / 3.7

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.35
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)

    # Crop - 5
    left = right
    top = height / 1.35
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)

    # Crop - 6
    left = width / 1.15
    top = height / 4
    right = width
    bottom = height / 1.35

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)

    # Crop - 7
    left = width / 1.35
    top = 0
    right = width
    bottom = height / 4.5

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (214, 500))
            train_image = cv2.resize(train_image, (214, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (287, 500))
            test_image = cv2.resize(test_image, (287, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 161))
            train_image = cv2.resize(train_image, (500, 161))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 138))
            train_image = cv2.resize(train_image, (500, 138))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 212))
            train_image = cv2.resize(train_image, (500, 212))
        elif position == 6:
            test_image = cv2.resize(test_image, (313, 500))
            train_image = cv2.resize(train_image, (313, 500))
        else:
            test_image = cv2.resize(test_image, (500, 188))
            train_image = cv2.resize(train_image, (500, 188))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111", "Crop_111111", "Crop_1111111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222", "Crop_222222", "Crop_2222222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333", "Crop_333333", "Crop_3333333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444", "Crop_444444", "Crop_4444444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555", "Crop_555555", "Crop_5555555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666", "Crop_666666", "Crop_6666666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777", "Crop_777777", "Crop_7777777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 7):
            matching_image(cv2.imread("static/Cutting/taka_100/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)
            # cv2.imshow("test_image - " + str(step) + " - " + str(j), tes_img[step][j])
            # cv2.imshow("train_image - " + str(step) + " - " + str(j), tr_img[step][j])
            # cv2.imshow("result - " + str(step) + " - " + str(j), res_img[step][j])

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
    #     print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))

    # print()

    # Checking note original or fake
    original_500_taka_list = [14.00, 22.00, 31.00, 22.00, 8.00, 5.00, 0.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_500_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def taka_200(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 7.5
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)

    # Crop - 2
    left = right
    top = 0
    right = width / 2.7
    bottom = height / 1.25

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)

    # Crop - 3
    left = right
    top = 0
    right = width / 1.45
    bottom = height / 3

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.45
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)

    # Crop - 5
    left = right
    top = height / 1.5
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)

    # Crop - 6
    left = width / 1.2
    top = height / 3
    right = width
    bottom = height / 1.45

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)

    # Crop - 7
    left = width / 1.35
    top = 0
    right = width
    bottom = height / 3.7

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (157, 500))
            train_image = cv2.resize(train_image, (157, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (348, 500))
            test_image = cv2.resize(test_image, (348, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 156))
            train_image = cv2.resize(train_image, (500, 156))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 156))
            train_image = cv2.resize(train_image, (500, 156))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 233))
            train_image = cv2.resize(train_image, (500, 233))
        elif position == 6:
            test_image = cv2.resize(test_image, (512, 470))
            train_image = cv2.resize(train_image, (512, 470))
        else:
            test_image = cv2.resize(test_image, (500, 222))
            train_image = cv2.resize(train_image, (500, 222))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111", "Crop_111111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222", "Crop_222222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333", "Crop_333333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444", "Crop_444444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555", "Crop_555555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666", "Crop_666666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777", "Crop_777777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 6):
            matching_image(cv2.imread("static/Cutting/taka_200/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)
            # cv2.imshow("test_image - " + str(step) + " - " + str(j), tes_img[step][j])
            # cv2.imshow("train_image - " + str(step) + " - " + str(j), tr_img[step][j])
            # cv2.imshow("result - " + str(step) + " - " + str(j), res_img[step][j])

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
    #     print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))

    # print()

    # Checking note original or fake
    original_200_taka_list = [17.00, 16.00, 8.00, 1.00, 9.00, 5.00, 8.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_200_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def taka_500(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 6.5
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)

    # Crop - 2
    left = right
    top = 0
    right = width / 2.7
    bottom = height / 1.25

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)

    # Crop - 3
    left = right
    top = 0
    right = width / 1.35
    bottom = height / 3.5

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.36
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)

    # Crop - 5
    left = right
    top = height / 1.3
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)

    # Crop - 6
    left = width / 1.2
    top = height / 3.5
    right = width
    bottom = height / 1.3

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)

    # Crop - 7
    left = width / 1.35
    top = 0
    right = width
    bottom = height / 4.5

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (180, 500))
            train_image = cv2.resize(train_image, (180, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (318, 500))
            test_image = cv2.resize(test_image, (318, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 164))
            train_image = cv2.resize(train_image, (500, 164))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 135))
            train_image = cv2.resize(train_image, (500, 135))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 185))
            train_image = cv2.resize(train_image, (500, 185))
        elif position == 6:
            test_image = cv2.resize(test_image, (405, 500))
            train_image = cv2.resize(train_image, (405, 500))
        else:
            test_image = cv2.resize(test_image, (500, 182))
            train_image = cv2.resize(train_image, (500, 182))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 5):
            matching_image(cv2.imread("static/Cutting/taka_500/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)
            

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
    #     print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))

    # print()

    # Checking note original or fake
    original_500_taka_list = [17.00, 17.00, 24.00, 11.00, 8.00, 6.00, 6.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_500_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def taka_1000(img_path):
    img = Image.open(img_path)
    crop_image = []
    width, height = img.size

    # Crop - 1
    left = 0
    top = 0
    right = width / 6.5
    bottom = height

    img_crop_1 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_1)

    # Crop - 2
    left = right
    top = 0
    right = width / 2.7
    bottom = height / 1.25

    img_crop_2 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_2)

    # Crop - 3
    left = right
    top = 0
    right = width / 1.35
    bottom = height / 3.5

    img_crop_3 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_3)

    # Crop - 4
    left = left
    top = height / 1.3
    right = width / 1.36
    bottom = height

    img_crop_4 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_4)

    # Crop - 5
    left = right
    top = height / 1.3
    right = width
    bottom = height

    img_crop_5 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_5)

    # Crop - 6
    left = width / 1.2
    top = height / 4
    right = width
    bottom = height / 1.3

    img_crop_6 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_6)

    # Crop - 7
    left = width / 1.35
    top = 0
    right = width
    bottom = height / 4.5

    img_crop_7 = img.crop((left, top, right, bottom))
    crop_image.append(img_crop_7)

    # Matching Image
    matching_points = [[] for i in range(0, 7)]
    tr_img = [[] for j in range(0, 7)]
    tes_img = [[] for k in range(0, 7)]
    res_img = [[] for m in range(0, 7)]

    def matching_image(train, test, position):
        train_image = train
        test_image = test

        if position == 1:
            test_image = cv2.resize(test_image, (176, 500))
            train_image = cv2.resize(train_image, (176, 500))
        elif position == 2:
            train_image = cv2.resize(train_image, (310, 500))
            test_image = cv2.resize(test_image, (310, 500))
        elif position == 3:
            test_image = cv2.resize(test_image, (500, 169))
            train_image = cv2.resize(train_image, (500, 169))
        elif position == 4:
            test_image = cv2.resize(test_image, (500, 138))
            train_image = cv2.resize(train_image, (500, 138))
        elif position == 5:
            test_image = cv2.resize(test_image, (500, 191))
            train_image = cv2.resize(train_image, (500, 191))
        elif position == 6:
            test_image = cv2.resize(test_image, (368, 500))
            train_image = cv2.resize(train_image, (368, 500))
        else:
            test_image = cv2.resize(test_image, (500, 187))
            train_image = cv2.resize(train_image, (500, 187))

        # check similarities between the 2 image
        sift = cv2.SIFT_create(nfeatures=1000)
        kp_1, desc_1 = sift.detectAndCompute(train_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)

        # print("Keypoints 1st Image: " + str(len(kp_1)))
        # print("Keypoints 2nd Image: " + str(len(kp_2)))

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)
        # print("Matches: ", len(matches))

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        # print("Good Points: ", len(good_points))
        # print("How Good it's the match: ", format((len(good_points) / number_keypoints) * 100, ".2f"), "%")
        # print()
        # print()
        # print()

        matching_points[position - 1].append((len(good_points) / number_keypoints) * 100)

        result = cv2.drawMatches(train_image, kp_1, test_image, kp_2, good_points[:100], None, flags=2)

        tr_img[position - 1].append(train_image)
        tes_img[position - 1].append(test_image)
        res_img[position - 1].append(result)

    train_image_name = [
        ["Crop_1", "Crop_11", "Crop_111", "Crop_1111", "Crop_11111", "Crop_111111"],
        ["Crop_2", "Crop_22", "Crop_222", "Crop_2222", "Crop_22222", "Crop_222222"],
        ["Crop_3", "Crop_33", "Crop_333", "Crop_3333", "Crop_33333", "Crop_333333"],
        ["Crop_4", "Crop_44", "Crop_444", "Crop_4444", "Crop_44444", "Crop_444444"],
        ["Crop_5", "Crop_55", "Crop_555", "Crop_5555", "Crop_55555", "Crop_555555"],
        ["Crop_6", "Crop_66", "Crop_666", "Crop_6666", "Crop_66666", "Crop_666666"],
        ["Crop_7", "Crop_77", "Crop_777", "Crop_7777", "Crop_77777", "Crop_777777"]
    ]

    def matching_processing(copy_image, step):
        for j in range(0, 6):
            matching_image(cv2.imread("static/Cutting/taka_1000/" + train_image_name[step][j] + ".jpg",
                                      cv2.IMREAD_GRAYSCALE),
                           copy_image, step + 1)
            

    for i in range(0, 7):
        matching_processing(cv2.cvtColor(np.array(crop_image[i]), cv2.COLOR_RGB2GRAY), i)

    # for i in range(len(matching_points)):
    #     print("Crop_" + str(i + 1) + ": ", matching_points[i], "\n")

    test_imagee = cv2.imread(img_path)
    test_imagee = cv2.resize(test_imagee, (600, 253))

    test_image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_image_gray = cv2.resize(test_image_gray, (600, 253))

    # print()

    # Checking note original or fake
    original_500_taka_list = [19.00, 26.00, 31.00, 13.00, 7.00, 4.00, 0.00]

    check_original_note = 0

    for i in range(0, 7):
        if min(matching_points[i]) >= original_500_taka_list[i]:
            check_original_note += 1

    if check_original_note == 7:
        # print("*** Original Note ***")
        return "Original Note"
    else:
        # print("*** Fake Note ***")
        return "Fake Note"

    cv2.waitKey(0)
    cv2.destroyAllWindows()

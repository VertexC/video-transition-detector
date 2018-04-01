from src.model.intersection import IntersectionDetector

def test_intersection_method():
    # video_path = 'media/left_wipe.avi'
    # video_path = 'media/video_1_horizontal_wipe.mp4'
    # video_path = 'media/video_2_horizontal_wipe.mp4'
    # video_path = 'media/video_1_vertical_wipe.mp4'
    # video_path = 'media/video_2_vertical_wipe.mp4'
    # video_path = 'media/video_3_down_wipe.mp4'
    # video_path = 'media/video_4_left_wipe.mp4'
    video_path = 'media/video_4_up_wipe.mp4'


    to_chroma = True

    model = IntersectionDetector()
    model.set_video(video_path)
    model.set_mode(to_chromatic=to_chroma)
    model.set_threshold(0.5)
    detect_result = model.detect()
    if detect_result == None:
        print("No transition detected")
    else:
        print(detect_result)
        model.show_result()



if __name__ == '__main__':
    test_intersection_method()
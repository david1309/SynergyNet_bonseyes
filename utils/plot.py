import cv2
import math


class plotUtils:
    def __init__(self, image):
        """
        Initialize the object
        
        :param image: matrix or path to an image
        :return: Plot util object for the image
        """
        if type(image) == str:
            self.image = cv2.imread(image)
        else:
            self.image = image

    def plot_bbox(self, bbox, color=(255, 0, 0), thickness=1):
        """
        Plot bounding box on the image
        
        :param bbox: Bounding box to plot_annotations, refer to the bounding_box definition from annotation catalog for more details
        :param color: Color to use for plotting the bbox (BGR)
        :param thickness: Line thickness to use for plotting the bounding box, must be integer
        """
        assert all(coord is not None for coord in bbox.values()), "No coordinate in argument 'bbox' shall be None."        
        x, y, w, h = round(float(bbox['x'])), round(float(bbox['y'])), round(float(bbox['w'])), round(float(bbox['h']))
        cv2.line(self.image, (x, y), (x + w, y), color=color, thickness=thickness)
        cv2.line(self.image, (x + w, y), (x + w, y + h), color=color, thickness=thickness)
        cv2.line(self.image, (x + w, y + h), (x, y + h), color=color, thickness=thickness)
        cv2.line(self.image, (x, y + h), (x, y), color=color, thickness=thickness)

    def plot_landmarks(self, landmarks, color=(255, 0, 0), radius=1, points_fill=True, index_offset=(5, -5),
                       index_line_thickness=1, index_font_scale=1.0):
        """
        Plot 2D landmarks on the image along with their indices

        :param landmarks: 2D Landmark array, refer to the definition from the annotation catalog for more details
        :param color: Color to use for plotting the landmarks (BGR)
        :param radius: Radius in pixels for the dots, must be an integer
        :param points_fill: If the dots must be filled with color or simply the outlines
        :param index_offset: Tuple specifying how far off the landmark index must be plotted from the dot, must be integer tuple
        :param index_line_thickness: Line thickness for writing the landmark indices, must be an integer
        :param index_font_scale: Font scale for index text
        """
        for index, landmark in enumerate(landmarks):
            assert all(coord is not None for coord in landmark.values()), "No coordinate in argument 'landmarks' shall be None."
            thickness = -1 if points_fill else 1
            point = (round(float(landmark['x'])), round(float(landmark['y'])))
            cv2.circle(self.image, point, radius=radius, color=color, thickness=thickness)
            self.plot_text(str(landmark['idx']), (point[0] + index_offset[0], point[1] + index_offset[1]),
                        color=color, thickness=index_line_thickness, font_scale=index_font_scale)

    def plot_text(self, text, target_loc,
                  color=(255, 0, 0),
                  font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=1):
        """
        Plot text on the image

        :param text: Text to be plotted
        :param target_loc: The (x, y) pixel location where the text must be plotted
        :param color: Color to be used for plotting the text (BGR)
        :param font: Font to be used for plotting the text, check cv2.FONT_* for more details and options
        :param font_scale: Font scale, which scales the base font size for used font
        :param thickness: Line thickness for plotting the text, must be an integer
        """
        assert all(coord is not None for coord in target_loc), "No coordinate in argument 'target_loc' shall be None."        
        cv2.putText(self.image, text,
                    org=target_loc,
                    fontFace=font, fontScale=font_scale, color=color, thickness=thickness)

    def plot_gaze_angles(self, eye_center, yaw, pitch, magnitude=100, color=(255, 0, 0), thickness=1):
        """
        Plot the gaze angles on eye center

        :param eye_center: (x,y) coordinate of eye center in pixels
        :param yaw: Gaze yaw angle in degrees, Subject's right is +ve Yaw angle
        :param pitch: Gaze pitch angle in degrees, Upwards is +ve Pitch angle
        :param magnitude: Magnitude of the vector to draw
        :param color: Color to be used for plotting (BGR)
        :param thickness: Line thickness for plotting the text, must be an integer
        """
        assert all(coord is not None for coord in eye_center), "No coordinate in argument 'eye_center' shall be None."
        assert yaw is not None, "Argument 'yaw' must not be None."
        assert pitch is not None, "Argument 'pitch' must not be None."
        
        target_x, target_y = round(math.sin(yaw) * math.cos(pitch) * magnitude), round(math.sin(pitch) * magnitude)
        target = (eye_center[0] - target_x, eye_center[1] - target_y)
        cv2.line(self.image, eye_center, target, color=color, thickness=thickness)

    def plot_headpose_angles(self, location, magnitude, yaw, pitch, roll, thickness=1, text_font_scale=1.0):
        """
        Plot the head pose angles

        :param location: Location of the pixel where to start the plotting (bottom left pixel)
        :param magnitude: Size of the plot_annotations (the plot_annotations will be using area of n x n pixels where n
        is the magnitude)
        :param yaw: Head pose yaw angle in degrees, Subject's right is +ve Yaw angle
        :param pitch: Head pose pitch angle in degrees, Upwards is +ve Pitch angle
        :param roll: Headpose roll angle in degrees, Tilting to subject's right is +ve Roll angle
        :param thickness: Line thickness to use for the plots
        """
        assert all(coord is not None for coord in location), "No coordinate in argument 'location' shall be None."
        assert magnitude is not None, "Argument 'magnitude' must not be None."
        assert yaw is not None, "Argument 'yaw' must not be None."
        assert pitch is not None, "Argument 'pitch' must not be None."
        assert roll is not None, "Argument 'roll' must not be None."
        
        yaw_color_line, yaw_color_dot = (100, 100, 255), (0, 0, 150)
        pitch_color_line, pitch_color_dot = (100, 255, 100), (0, 150, 0)
        roll_color_line, roll_color_dot = (255, 100, 100), (150, 0, 0)

        center = (location[0] + (magnitude // 2), location[1] - (magnitude // 2))
        top = (location[0], location[1] - magnitude)
        bottom = (location[0] + magnitude, location[1])

        # Yaw
        cv2.line(self.image, (top[0], center[1]), (bottom[0], center[1]), color=yaw_color_line, thickness=thickness)
        yaw_point = (center[0] - round((magnitude / 2) * (float(yaw) / 90)), center[1])

        # Pitch
        cv2.line(self.image, (center[0], top[1]), (center[0], bottom[1]), color=pitch_color_line, thickness=thickness)
        pitch_point = (center[0], center[1] - round((magnitude / 2) * (float(pitch) / 90)))

        # Roll
        radius = magnitude // 2
        axes = (radius, radius)
        angle = 0
        start_angle = 180
        end_angle = 360
        cv2.ellipse(self.image, center=center, axes=axes, angle=angle,
                    startAngle=start_angle,
                    endAngle=end_angle,
                    color=roll_color_line,
                    thickness=thickness)
        roll_point = (center[0] - round(math.sin(math.radians(roll)) * radius),
                      center[1] - round(math.cos(math.radians(roll)) * radius))

        cv2.circle(self.image, yaw_point, thickness + 1, yaw_color_dot, -1)
        cv2.circle(self.image, pitch_point, thickness + 1, pitch_color_dot, -1)
        cv2.circle(self.image, roll_point, thickness + 1, roll_color_dot, -1)

        text_thickness = thickness - 1 if thickness - 1 >= 1 else 1
        self.plot_text('Yaw', (yaw_point[0], yaw_point[1] - thickness - 5), yaw_color_dot, font_scale=text_font_scale,
                       thickness=text_thickness)
        self.plot_text('Pitch', (pitch_point[0] + thickness + 5, pitch_point[1]), pitch_color_dot,
                       font_scale=text_font_scale, thickness=text_thickness)
        self.plot_text('Roll', (roll_point[0], roll_point[1] + thickness + 10), roll_color_dot,
                       font_scale=text_font_scale, thickness=text_thickness)

    def save(self, output_path):
        """
        Save the image to disk after plotting everything

        :param output_path: The path on disk where the image must be saved
        """
        cv2.imwrite(output_path, self.image)

"""Script file to create HTML page with several images."""
import os
import numpy as np
import skimage.io as skio
import skimage.color as skicol


class ImageFeatureExtractor:
    """Utility to extract features from a directory containing images."""

    def __init__(self, directory='Wang_Data'):
        """Initialize this object with a directory and 1000 images."""
        self.directory = directory
        self._image_names = []
        for i in range(1000):
            self._image_names.append(str(i) + '.jpg')

    def load_images(self, list_indices=None, start=None, end=None):
        """Load images in memory."""
        if start is None and list_indices is None:
            start = 0
        if end is None and list_indices is None:
            end = len(self._image_names)
        if list_indices is None:
            assert start >= 0
            assert start < end
            assert end <= len(self._image_names)
            list_indices = np.arange(start, end)

        self.image_indices = []
        self.images = []
        self.image_names = []
        for i, image_name in enumerate(self._image_names):
            if i in list_indices:
                image_filename = os.path.join(self.directory, image_name)
                image = skio.imread(image_filename)
                self.image_indices.append(i)
                self.images.append(image)
                self.image_names.append(image_name)
        print(len(self.images), 'images loaded!')

    def extract_histogram(self, bins=10):
        """Extract grey intensity histogram."""
        assert len(self.images) > 0, 'No images loaded! Did you call ' \
                                     'load_images() ?'
        histograms = []
        for image in self.images:
            grey = skicol.rgb2grey(image)
            hist_values, bins = np.histogram(grey, range=(0, 1), bins=bins)
            histograms.append(hist_values)
        histograms = np.array(histograms)
        histograms = histograms.astype('float')
        return histograms

    def extract_color_histogram(self, bins=10):
        """Extract color intensity histogram."""
        assert len(self.images) > 0, 'No images loaded! Did you call ' \
                                     'load_images() ?'
        color_histograms = []
        for image in self.images:
            histograms = []
            for color in range(image.shape[2]):
                band = image[:, :, color].reshape(-1)
                hist_values, bins = np.histogram(band, range=(0, 255),
                                                 bins=bins)
                histograms += list(hist_values)
            color_histograms.append(histograms)
        color_histograms = np.array(color_histograms)
        color_histograms = color_histograms.astype('float')
        return color_histograms

    def extract_hue_histogram(self, bins=10):
        """Extract hue histogram."""
        assert len(self.images) > 0, 'No images loaded! Did you call ' \
                                     'load_images() ?'
        hue_histograms = []
        for image in self.images:
            hue = skicol.rgb2hsv(image)[:, :, 0].reshape(-1)
            hist_values, bins = np.histogram(hue, range=(0, 1), bins=bins)
            hue_histograms.append(hist_values)
        hue_histograms = np.array(hue_histograms)
        hue_histograms = hue_histograms.astype('float')
        return hue_histograms

    def to_html(self, file_name, kohonen_map):
        """Write images contained in a Kohonen map to an HTML file."""
        if file_name[-5:] != '.html':
            print('Not html file... appending HTML extension!')
            file_name += '.html'
        with open(file_name, 'w') as html_file:
            html_content = """<html>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>
            <style type="text/css">
                #overlay {
                    position: absolute;
                    top: 40;
                    left: 40;
                    width: 400px;
                    height: 400px;
                    background-color: #000;
                }
            </style>
            <script type="text/javascript">
                $(document).ready(function () {
                    $('#overlay').hide();
                });

                function close() {
                    $('#overlay').hide();
                }

                function show_images(imgs) {
                    if (imgs.length == 0) {
                        alert("No other images in that neuron!");
                    } else {
                        var div = $('#overlay');
                        div.empty();
                        div.css('text-align', 'center')

                        var btn = $('<button>Close</button><br/><br/>');
                        btn.on('click', close);
                        div.append(btn);

                        for (var i = 0; i < imgs.length; ++i) {
                            var img = imgs[i];
                            div.append($('<img src="' + img + '" style="width:100px;height:100px;"/>'));
                            div.show();
                        }
                    }
                }
            </script>

            <div id="container">
                <table>
            """ # noqa
            html_file.write(html_content)
            for i in range(kohonen_map._map.shape[0]):
                html_file.write('<tr>')
                for j in range(kohonen_map._map.shape[1]):
                    html_file.write('<td>')
                    key = str(i) + ',' + str(j)
                    sample_ids = kohonen_map.samples_dict[key]
                    if len(sample_ids) > 0:
                        style = 'style=\"width:100px;height:100px;\"'
                        images_list = []
                        for index, sample_id in enumerate(sample_ids):
                            image_index = self.image_indices[sample_id]
                            image_name = str(image_index) + '.jpg'
                            image_file = os.path.join(self.directory,
                                                      image_name)
                            if index == 0:
                                src = 'src="' + image_file + '"'
                                alt = 'alt="' + image_file + '"'
                            else:
                                images_list.append("'" + image_file + "'")
                        images_list = ','.join(images_list)
                        onclick = 'onclick="show_images([' + images_list + \
                                  '])"'
                        img = '<img ' + src + ' ' + alt + ' ' + style + ' ' + \
                              onclick + '>'
                        html_file.write(img)
                    else:
                        alt = 'alt=\"No image\"'
                        style = 'style=\"width:100px;height:100px;\"'
                        #html_file.write(img)
                    html_file.write('</td>')
                html_file.write('</tr>')
            html_content = """</table>
            </div>

            <div id="overlay">
            </div>
            </body>
            </html>"""
            html_file.write(html_content)

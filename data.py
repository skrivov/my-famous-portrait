import os

# Style Images Data
style_images_files = ['mona-lisa.jpg', 'picasso.jpg', 'helena-wierzbicki.jpg', 'vangogh.jpg', 'woman.jpeg']

style_images_names = ['Mona Lisa', 'Picasso Self Portrait', 'Helena Wierzbicki', 'Vangogh Self Portrait', 'Picasso, Woman with Hat']

styles_path = 'styles'

style_images_dict = {name: os.path.join(styles_path, file) for name, file in zip(style_images_names, style_images_files)}

# Celeb Images Data
celeb_images_files = ['hermione.jpeg', 'potter.jpg', 'albus-dumbledore.jpg' ]

celeb_images_names = ['Hermione', 'Harry Potter', 'Albus Dumbledore']

celebs_path = 'celebs'

celebs_images_dict = {name: os.path.join(celebs_path, file) for name, file in zip(celeb_images_names, celeb_images_files)}

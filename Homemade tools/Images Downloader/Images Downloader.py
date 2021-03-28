from simple_image_download import simple_image_download as simp

response = simp.simple_image_download()

words = """arvore de Imperatriz, galho arvore de Imperatriz,
Manaca de Jardim, galho Manaca de Jardim, Quaresmeira, galho Quaresmeira, Falsa Murta,
galho Falsa Murta, Santa Barba, galho Santa Barba, Sombreiro, galho Sombreiro, Cassia-do-nordeste, galho Cassia-do-nordeste,
Flamboyant, galho Flamboyant, Oiti, galho Oiti"""

response.download(keywords=words, limit=500, extensions={'.jpeg', '.jpg', '.gif', '.png', '.ico', '.tff', '.psd', '.eps', '.ai', '.raw', '.indd', '.pdf', '.webp', '.dib', '.svg ', '.svgz', '.pic', '.bmp', '.tif', '.jfif'})

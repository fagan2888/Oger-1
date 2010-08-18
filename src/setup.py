from distutils.core import setup

short_description = "Engine Toolbox"

long_description = """
The Engine is a Python toolbox for rapidly building, training and evaluating hierarchical learning architectures on large datasets. It builds functionality on top of the Modular toolkit for Data Processing (MDP).  The Engine builds functionality on top of MDP, such as:
 - Cross-validation of datasets
 - Grid-searching large parameter spaces
 - Processing of temporal datasets
 - Gradient-based training of deep learning architectures
 - Interface to the Speech Processing, Recognition, and Automatic Annotation Kit (SPRAAK) 

 In addition, several additional MDP nodes are provided by the Engine, such as a:
 - Reservoir node
 - Leaky reservoir node
 - Ridge regression node
 - Conditional Restricted Boltzmann Machine (CRBM) node
"""

classifiers = ["Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]


setup(name = 'Engine', version = '0.2',
      author = "Philemon Brakel, Martin Fiers, Fiontann O'Donnell, Benjamin Schrauwen and David Verstraeten",
      author_email = 'pbpop3@gmail.com, mfiers@intec.ugent.be, fodonnel@elis.ugent.be, benjamin.schrauwen@elis.ugent.be, david.verstraeten@elis.ugent.be',
      maintainer = "Philemon Brakel, Martin Fiers, Fiontann O'Donnell, Benjamin Schrauwen and David Verstraeten",
      maintainer_email = 'pbpop3@gmail.com, mfiers@intec.ugent.be, fodonnel@elis.ugent.be, benjamin.schrauwen@elis.ugent.be, david.verstraeten@elis.ugent.be',
      platforms = ["Any"],
      url = 'http://organic.elis.ugent.be',
      download_url = 'http://organic.elis.ugent.be/Engine',
      description = short_description,
      long_description = long_description,
      classifiers = classifiers,
      packages = ['Engine', 'Engine.datasets', 'Engine.evaluation', 'Engine.gradient', 'Engine.utils', 'Engine.nodes', 'Engine.parallel',
                  'mdp', 'mdp.nodes', 'mdp.utils', 'mdp.hinet', 'mdp.test', 'mdp.demo', 'mdp.graph', 'mdp.contrib', 'mdp.parallel'],
      package_data={'mdp': ['hinet/*.css', 'utils/*.css'], 'Engine': ['examples/*.py']},
      )

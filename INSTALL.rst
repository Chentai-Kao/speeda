install git
----------------------------------------------
sudo apt-get install git

generate ssh-key (to clone Git)
----------------------------------------------
cd ~/.ssh && ssh-keygen
cat id_rsa.pub | xclip
git config --global user.name "ru6xul6"
git config --global user.email "ru6xul6@gmail.com"

setup environment
----------------------------------------------
(for color terminal prompt, uncomment "force_color_prompt" in ~/.bashrc)
(copy over my .vimrc)

clone speeda
----------------------------------------------
git clone git@github.com:Cobra-Kao/speeda.git

Install pip
----------------------------------------------
curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | sudo python2.7

Install python development tool
----------------------------------------------
sudo apt-get install libxml2-dev libxslt-dev python-dev

Install lxml (python xml tool)
----------------------------------------------
sudo pip install lxml

Install numpy
----------------------------------------------
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

Install kdenlive and MLT (video editing framework)
(There will be warning about libav and ffmpeg, that's fine, keep going)
----------------------------------------------
sudo add-apt-repository ppa:sunab/kdenlive-release && sudo apt-get update && sudo apt-get install kdenlive

Install sox (speed up audio file)
----------------------------------------------
sudo apt-get install sox

Install Matplotlib 1.4.2 (apt-get version is outdated)
----------------------------------------------
sudo apt-get install pkg-config
sudo apt-get install libfreetype6-dev
sudo easy_install http://cheeseshop.python.org/packages/source/p/pyparsing/pyparsing-2.0.1.tar.gz
wget https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.4.2/matplotlib-1.4.2.tar.gz
tar xvf matplotlib-1.4.2.tar.gz
cd matplotlib-1.4.2
sudo python setup.py build
sudo python setup.py install

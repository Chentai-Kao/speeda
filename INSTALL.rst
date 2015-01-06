Upgrade to Ubuntu 14.04 (newer package support, like yasm should be 1.2.0)
----------------------------------------------
do-release-upgrade

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

Install MLT
----------------------------------------------
sudo apt-get install git automake autoconf libtool intltool g++ yasm libmp3lame-dev libgavl-dev libsamplerate-dev libxml2-dev ladspa-sdk libjack-dev libsox-dev libsdl-dev libgtk2.0-dev liboil-dev libsoup2.4-dev libqt4-dev libexif-dev libvdpau-dev libdv-dev libtheora-dev libvorbis-dev subversion cmake kdelibs5-dev libqjson-dev libqimageblitz-dev recordmydesktop dvgrab dvdauthor genisoimage xine-ui libeigen3-dev xutils-dev libegl1-mesa-dev libfftw3-dev
cd ~
mkdir kdenlive
cd kdenlive
wget http://github.com/mltframework/mlt-scripts/raw/master/build/build-kdenlive.sh
chmod +x build-kdenlive.sh
./build-kdenlive.sh

Install kdenlive and MLT (ignore warning about libav and ffmpeg)
----------------------------------------------
sudo add-apt-repository ppa:sunab/kdenlive-release && sudo apt-get update && sudo apt-get install kdenlive

Install sox (speed up audio file)
----------------------------------------------
sudo apt-get install sox

Install mp3 codec (for rendering)
----------------------------------------------
sudo apt-get install libavcodec-extra-53

Install ffmpeg
----------------------------------------------
sudo add-apt-repository ppa:mc3man/trusty-media && sudo apt-get update && sudo apt-get install ffmpeg

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







ERROR Problem
Render aborted when "bash test.sh"


# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2019 Grey Li
    :license: MIT, see LICENSE for more details.
"""
import os
import uuid

from flask import Flask, render_template, flash, redirect, url_for, request, send_from_directory, session
from flask_ckeditor import CKEditor, upload_success, upload_fail
from flask_dropzone import Dropzone
from flask_wtf.csrf import validate_csrf
from wtforms import ValidationError

from forms import LoginForm, FortyTwoForm, NewPostForm, UploadForm, MultiUploadForm, SigninForm, \
    RegisterForm, SigninForm2, RegisterForm2, RichTextForm

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'secret string')

# Custom config
app.config['UPLOAD_PATH'] = os.path.join(app.root_path, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = ['png', 'jpg', 'jpeg', 'gif']

# Flask config
# set request body's max length
# app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # 3Mb

# Flask-CKEditor config
app.config['CKEDITOR_SERVE_LOCAL'] = True
app.config['CKEDITOR_FILE_UPLOADER'] = 'upload_for_ckeditor'

# Flask-Dropzone config
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image'
app.config['DROPZONE_MAX_FILE_SIZE'] = 3
app.config['DROPZONE_MAX_FILES'] = 30

ckeditor = CKEditor(app)
dropzone = Dropzone(app)


#@app.route('/', methods=['GET', 'POST'])
#def index():
#    return render_template('index2.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = random_filename(f.filename)
        !python track.py --source filename
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        flash('Upload success.')
        session['filenames'] = [filename]
        return redirect(url_for('show_images'))
    return render_template('index2.html', form=form)


@app.route('/html', methods=['GET', 'POST'])
def html():
    form = LoginForm()
    if request.method == 'POST':
        username = request.form.get('username')
        flash('Welcome home, %s!' % username)
        return redirect(url_for('index'))
    return render_template('pure_html.html')


@app.route('/basic', methods=['GET', 'POST'])
def basic():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        flash('Welcome home, %s!' % username)
        return redirect(url_for('index'))
    return render_template('basic.html', form=form)


@app.route('/bootstrap', methods=['GET', 'POST'])
def bootstrap():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        flash('Welcome home, %s!' % username)
        return redirect(url_for('index'))
    return render_template('bootstrap.html', form=form)


@app.route('/custom-validator', methods=['GET', 'POST'])
def custom_validator():
    form = FortyTwoForm()
    if form.validate_on_submit():
        flash('Bingo!')
        return redirect(url_for('index'))
    return render_template('custom_validator.html', form=form)


@app.route('/uploads/<path:filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/uploaded-images')
def show_images():
    return render_template('uploaded.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = random_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        flash('Upload success.')
        session['filenames'] = [filename]
        return redirect(url_for('show_images'))
    return render_template('upload.html', form=form)

@app.route('/handle-signin', methods=['POST'])
def handle_signin():
    signin_form = SigninForm2()
    register_form = RegisterForm2()

    if signin_form.validate_on_submit():
        username = signin_form.username.data
        flash('%s, you just submit the Signin Form.' % username)
        return redirect(url_for('index'))

    return render_template('2form2view.html', signin_form=signin_form, register_form=register_form)


@app.route('/handle-register', methods=['POST'])
def handle_register():
    signin_form = SigninForm2()
    register_form = RegisterForm2()

    if register_form.validate_on_submit():
        username = register_form.username.data
        flash('%s, you just submit the Register Form.' % username)
        return redirect(url_for('index'))
    return render_template('2form2view.html', signin_form=signin_form, register_form=register_form)


@app.route('/ckeditor', methods=['GET', 'POST'])
def integrate_ckeditor():
    form = RichTextForm()
    if form.validate_on_submit():
        title = form.title.data
        body = form.body.data
        flash('Your post is published!')
        return render_template('post.html', title=title, body=body)
    return render_template('ckeditor.html', form=form)


# handle image upload for ckeditor
@app.route('/upload-ck', methods=['POST'])
def upload_for_ckeditor():
    f = request.files.get('upload')
    if not allowed_file(f.filename):
        return upload_fail('Image only!')
    f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
    url = url_for('get_file', filename=f.filename)
    return upload_success(url, f.filename)

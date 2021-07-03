from pathlib import Path
from datetime import datetime

from flask.helpers import safe_join
from webapp import (
    app, IMG_COMPOSITION_CONFIG, OBJ_DETECTION_CONFIG
)
from flask import (
    url_for, request, render_template,
    send_from_directory, Flask, Response, make_response
)
from werkzeug.utils import secure_filename
import werkzeug
import cv2
import threading
from filelock import FileLock

from obj_detect import show_inference
from image_composition import composite_image

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

 #制限を追加
limiter = Limiter(app, key_func=get_remote_address, default_limits=["50 per minute"])


def is_limited_mode():
    light_mode = app.config.get('LIMITED')
    if light_mode:
        print("[INFO] LIMITED MODE ENABLED !")
    return bool(light_mode)


def prepare_response(data):
    response = make_response(data)
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


#GETの処理
@app.route('/', methods=['GET'])
@limiter.limit("10 per minute") # 制限を追加
def up_get():
    _light = "_light" if is_limited_mode() else ""
    response_body = render_template(f'index{_light}.j2', message = '画像を選択しよう．(^^)/')
    response = prepare_response(response_body)
    return response


@app.route('/', methods=['POST'])
@limiter.limit("10 per minute") # 制限を追加
def upload():
    _light = "_light" if is_limited_mode() else ""
    response_body = render_template(f'index{_light}.j2')
    response = prepare_response(response_body)
    return response


##############
#拡張子判別関数
##############
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpeg', 'jpg', 'JPEG', 'JPG']) # アップロードされる拡張子の制限
def allowed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
##############

@app.route('/upload_file', methods=['POST'])
@limiter.limit("20 per minute")  # 制限を追加
def upload_file():
    _light = "_light" if is_limited_mode() else ""

    def change_files(image_names, target_dir):
        for filename in image_names:
            in_filepath = str(target_dir / filename)
            out_filepath = str(target_dir / f"edited_{filename}")
            print(f"[INFO] conversion of {in_filepath} started.")
            image_in = cv2.imread(in_filepath)
            inference_result = show_inference(image_in.copy())
            image_out = composite_image(image_in, inference_result, IMG_COMPOSITION_CONFIG)
            with FileLock(out_filepath + ".lock"):
                cv2.imwrite(out_filepath, image_out)
            print(f"[INFO] conversion of {in_filepath} finished.")


    #変換前の画像がアップロードされているかどうかの判定
    if 'file' not in request.files:
        #response_body = render_template('index.j2', message = '画像が選択されていません．(T_T)')
        response_body = render_template(f'index{_light}.j2', message = '画像が選択されていません．(T_T)')
        response = prepare_response(response_body)
        return response

    files = request.files.getlist('file')

    image_names=[]
    image_in_urls = []
    image_out_urls = []
    subdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    target_dir = app.config['UPLOAD_FOLDER'] / subdir
    Path.mkdir(target_dir, exist_ok=True)

    max_num_files = 2 if is_limited_mode() else 5
    if len(files) > max_num_files:
        message = f'アップロードできるのは {max_num_files} ファイルまでです．'
        response_body = render_template(f'index{_light}.j2', message=message)
        response = prepare_response(response_body)
        return response

    for file in files:
        if file.filename == '':
            response_body = render_template(f'index{_light}.j2', message = '画像が選択されていません．(T_T)')
            response = prepare_response(response_body)
            return response


        #正しい拡張子のファイルがアップロードされた場合
        if file and allowed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）

            filename = secure_filename(file.filename)
            filepath = str(app.config['UPLOAD_FOLDER'] / subdir / filename)
            file.save(filepath)
            image_in_urls.append(url_for('uploaded_file', subdir=subdir, filename=filename))
            image_out_urls.append(url_for('uploaded_file', subdir=subdir, filename=f"edited_{filename}"))
            image_names.append(filename)


    #正しい拡張子のファイルがアップロードされた場合
    if all(map(lambda f: allowed_file(f.filename), files)):
        t = threading.Thread(target=change_files, args=(image_names, target_dir))
        t.start()
        kwargs = dict(
            title = 'Form Sample(post)',
            is_image_uploaded=True,
            image_in_urls=image_in_urls,
            image_out_urls=image_out_urls,
            image_names=image_names
        )
        response_body = render_template(f'result{_light}.j2', **kwargs)
        response = prepare_response(response_body)
        return response
    else: #指定した拡張子とは異なる拡張子のファイルがアップロードされた場合
        response_body = render_template(f'index{_light}.j2', message = '画像ではないファイルが選択されました．画像 (.png, .PNG, .jpeg, .jpg, .JPEG, .JPG) を選択してください．(T_T)')
        response = prepare_response(response_body)
        return response


@app.route('/uploads/<subdir>/<filename>', methods=["GET"])
@limiter.limit("50 per minute") # 制限を追加
def uploaded_file(subdir, filename):
    dirpath = safe_join(app.config['UPLOAD_FOLDER'], subdir)
    with FileLock(safe_join(dirpath, filename) + ".lock"):
        response = send_from_directory(dirpath, filename)
    return response



@app.route('/favicon.ico')
def favicon():
    img_dir = safe_join(app.static_folder, "image")
    return send_from_directory(img_dir, "favicon.ico")


##エラー処理
@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    message = 'アップロードされた画像が大きすぎます．(T T)'
    _light = "_light" if is_limited_mode() else ""
    response_body = render_template(f'index{_light}.j2', message=message)
    response = prepare_response(response_body)
    return response


@app.errorhandler(werkzeug.exceptions.TooManyRequests)
def handle_too_many_requests(error):
    _light = "_light" if is_limited_mode() else ""
    response_body = render_template(f'too_many_requests{_light}.j2')
    response = prepare_response(response_body)
    return response

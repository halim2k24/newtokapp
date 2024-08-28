# language.py
class Language:
    def __init__(self):
        self.languages = {
            'en': {
                'home': 'Home',
                'camera': 'Camera',
                'capture_image': 'Capture Image',
                'capture_video': 'Capture Video',
                'upload_image': 'Upload Image',
                'picking_settings': 'Picking Settings',
                'placing_settings': 'Placing Settings',
                'lan': '日本語',
                'exit': 'Exit',
                'task_panel': 'Task Panel',
                'image_view': 'Image View',
                'properties': 'Properties',
                'test_button': 'Test Button',
                'add_model_button': 'Add new model',
                'status': 'Status',
                'ok': 'OK',
                'ng': 'NG',
                'new_model_name': 'New Model Name',
                'name': 'Model Name',
                'select_shape': 'Select Shape',
                'save': 'Save',
                'width': 'Width',
                'height': 'Height',
                'matching': 'Matching %',
                'angle': 'Angle',
                'radius': 'Radius',
                'detection_order': 'Detection Order',
                'count': 'Count',
                'musk': 'Musk',
                'picking_condition': 'Picking Condition',
                'picking_point': 'Picking Point',
                'picking_point_shape': 'Picking Point Shape',
                'picking_settings_title': 'Picking Settings',
                'num_picking_objects': 'Number of Picking Object',
                'matching_percentage': 'Matching %',
                'picking_order': 'Picking Order',
                'placing_settings_title': 'Placing Settings',
                'placing_point_x': 'Placing Point X',
                'placing_point_y': 'Placing Point Y',
                'placing_angle': 'Placing Angle',
                'update': 'Update',
                'rectangle': 'Rectangle',
                'circle': 'Circle',
                'polygon': 'Polygon',
                'ring': 'Ring',
                'please_enter_valid_number': 'Please enter a valid number',
                'number_of_picking_object': 'Picking Object Count',
                'cannot_be_empty': 'cannot be empty',
                'delete_confirmation_title': 'Delete Model',
                'delete_confirmation_message': 'Are you sure you want to delete the model "{model_name}"?',
                'model_name_label': 'Model Name:',
                'center_x': 'Center X',
                'center_y': 'Center Y',
                'rotation_angle': 'Rotation Angle',
                'box_size': 'Box Size',
            },
            'jp': {
                'home': 'ホーム',
                'camera': 'カメラ',
                'capture_image': '画像をキャプチャ',
                'capture_video': 'ビデオをキャプチャ',
                'upload_image': '画像をアップロード',
                'picking_settings': 'ピッキング設定',
                'placing_settings': '配置設定',
                'lan': 'English',
                'exit': '出口',
                'task_panel': 'タスクパネル',
                'image_view': '画像表示',
                'properties': 'プロパティ',
                'test_button': 'テストボタン',
                'add_model_button': '新しいモデルを追加',
                'status': '状態',
                'ok': 'OK',
                'ng': 'NG',
                'new_model_name': '新しいモデル名',
                'name': 'モデル 名前 ',
                'save': '保存',
                'width': '幅',
                'height': '身長',
                'matching': 'マッチング %',
                'angle': '角度',
                'radius': '半径',
                'detection_order': '検出順序',
                'count': 'カウント',
                'select_shape': '図形を選択',
                'musk': 'マスク',
                'picking_condition': 'ピッキング条件',
                'picking_point': 'ピッキングポイント',
                'picking_point_shape': 'ピッキングポイントの形状',
                'picking_settings_title': 'ピッキング設定',
                'num_picking_objects': 'ピッキングオブジェクトの数',
                'matching_percentage': '一致率',
                'picking_order': 'ピッキング順序',
                'placing_settings_title': '配置設定',
                'placing_point_x': '配置ポイントX',
                'placing_point_y': '配置ポイントY',
                'placing_angle': '配置角度',
                'update': '更新',
                'rectangle': '矩形',
                'circle': '丸',
                'polygon': 'ポリゴン',
                'ring': '指輪',
                'please_enter_valid_number': '有効な数値を入力してください',
                'number_of_picking_object': 'オブジェクトの選択 カウント',
                'cannot_be_empty': 'を入力してください',
                'delete_confirmation_title': 'モデルを削除',
                'delete_confirmation_message': 'モデル "{model_name}" を削除してもよろしいですか？',
                'model_name_label': 'モデル名:',
                'center_x': '中心X',
                'center_y': '中心Y',
                'rotation_angle': '回転角度',

            }
        }
        self.current_language = 'en'

    def set_language(self, lang):
        if lang in self.languages:
            self.current_language = lang

    def get_current_language(self):
        return self.current_language

    def translate(self, key):
        return self.languages[self.current_language].get(key, key)

    def switch_language(self):
        self.current_language = 'jp' if self.current_language == 'en' else 'en'

    def get_opposite_language(self):
        return 'jp' if self.current_language == 'en' else 'en'


language = Language()

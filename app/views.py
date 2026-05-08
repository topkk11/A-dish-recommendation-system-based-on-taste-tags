from flask import render_template, request, redirect, url_for, flash
from app import app
from app.data import store
from config import RECOMMEND_TOP_N


@app.route('/')
def index():
    users = []
    if store.all_outputs_exist():
        users_df = store.load_users()
        users = users_df.to_dict('records')
    return render_template('index.html', users=users)


@app.route('/user/<user_id>')
def user_detail(user_id):
    if not store.all_outputs_exist():
        flash('暂无数据，请先导入评论数据', 'warning')
        return redirect(url_for('index'))

    users_df = store.load_users()
    matches = users_df[users_df['user_id'] == user_id]
    if matches.empty:
        flash(f'用户 {user_id} 不存在', 'danger')
        return redirect(url_for('index'))

    reviews_df = store.load_reviews()
    user = matches.iloc[0].to_dict()
    user_reviews = reviews_df[reviews_df['user_id'] == user_id].to_dict('records')
    return render_template('user_detail.html', user=user, reviews=user_reviews)


@app.route('/dish/<dish_id>')
def dish_detail(dish_id):
    if not store.all_outputs_exist():
        flash('暂无数据，请先导入评论数据', 'warning')
        return redirect(url_for('index'))

    dishes_df = store.load_dishes()
    matches = dishes_df[dishes_df['dish_id'] == dish_id]
    if matches.empty:
        flash(f'菜品 {dish_id} 不存在', 'danger')
        return redirect(url_for('index'))

    reviews_df = store.load_reviews()
    dish = matches.iloc[0].to_dict()
    dish_reviews = reviews_df[reviews_df['dish_id'] == dish_id].to_dict('records')
    return render_template('dish_detail.html', dish=dish, reviews=dish_reviews)


@app.route('/recommend/<user_id>')
def recommend(user_id):
    if not store.all_outputs_exist():
        flash('暂无数据，请先导入评论数据', 'warning')
        return redirect(url_for('index'))

    users_df = store.load_users()
    matches = users_df[users_df['user_id'] == user_id]
    if matches.empty:
        flash(f'用户 {user_id} 不存在', 'danger')
        return redirect(url_for('index'))

    from app.services.recommend_service import recommend as do_recommend
    user = matches.iloc[0].to_dict()
    results = do_recommend(user_id, top_n=RECOMMEND_TOP_N)
    return render_template('recommend.html', user=user, results=results)


@app.route('/import', methods=['GET', 'POST'])
def import_data():
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or file.filename == '':
            flash('请选择一个CSV文件', 'danger')
            return redirect(url_for('import_data'))

        if not file.filename.endswith('.csv'):
            flash('请上传CSV格式的文件', 'danger')
            return redirect(url_for('import_data'))

        from config import DATA_DIR
        import os
        filepath = os.path.join(DATA_DIR, 'reviews.csv')
        file.save(filepath)

        try:
            from app.nlp.pipeline import run_full_pipeline
            summary = run_full_pipeline(filepath)
            flash(
                f'导入成功！共 {summary["num_reviews"]} 条评论，'
                f'{summary["num_dishes"]} 个菜品，{summary["num_users"]} 个用户',
                'success'
            )
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'处理失败：{str(e)}', 'danger')
            return redirect(url_for('import_data'))

    return render_template('import.html')

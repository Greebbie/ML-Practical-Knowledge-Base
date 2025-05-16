from api.index import app

if __name__ == "__main__":
    # 本地开发服务器
    app.run(debug=True, port=5000) 
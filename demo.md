① POST /api/chat-messages web/completion.py → chat/app_generator.py → chat/app_runner.py 
② GET  /api/passport web/passport.py → web/wraps.py 
③ GET  /api/parameters web/app.py 
④ GET  /console/api/apps console/app/app.py → console/wraps.py → app_service.py 
⑤ POST /v1/chat-messages service_api/app/completion.py → service_api/wraps.py 
⑥ POST /workflows/draft/run console/app/workflow.py → workflow/app_generator.py 
⑦ POST /api/files/upload web/files.py → services/file_service.py

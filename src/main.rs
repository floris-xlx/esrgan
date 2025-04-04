use actix_cors::Cors;
use actix_multipart::Field;
use actix_multipart::Multipart;
use actix_web::http::header::ContentDisposition;
use actix_web::{web, web::Data, App, HttpResponse, HttpServer, Responder};
use futures_util::stream::StreamExt as _;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Error;
use std::io::Result as ioResult;
use std::path::Path;
use tokio::sync::Mutex;
use tracing::{error, info, trace, warn};
use uuid::Uuid;
use tokio::sync::MutexGuard;

use tracing_subscriber::EnvFilter;

struct AppState {
    cache_dir: String,
    request_status: Mutex<HashMap<String, String>>,
}

async fn upscale_image_post(data: Data<Mutex<AppState>>, mut payload: Multipart) -> impl Responder {
    info!("Received request to upscale image");
    let cache_dir: String = data.lock().await.cache_dir.clone();
    let request_id: String = Uuid::new_v4().to_string();
    info!("Started computing request ID: {}", request_id);
    let mut image_name: String = String::new();
    let mut original_path: String = String::new();
    let mut upscaled_path: String = String::new();

    let app_state: MutexGuard<'_, AppState> = data.lock().await;
    let mut status_map: MutexGuard<'_, HashMap<String, String>> =
        app_state.request_status.lock().await;
    status_map.insert(request_id.clone(), "Processing".to_string());

    while let Some(item) = payload.next().await {
        let mut field: Field = match item {
            Ok(field) => field,
            Err(e) => {
                error!("Error processing multipart item: {:?}", e);
                status_map.insert(request_id.clone(), "Error".to_string());
                return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
            }
        };
        let content_disposition: &ContentDisposition = match field.content_disposition() {
            Some(cd) => cd,
            None => {
                error!("Content disposition not found");
                status_map.insert(request_id.clone(), "Error".to_string());
                return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
            }
        };
        image_name = match content_disposition.get_filename() {
            Some(name) => name.to_string(),
            None => {
                error!("Filename not found in content disposition");
                status_map.insert(request_id.clone(), "Error".to_string());
                return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
            }
        };
        info!("Processing file: {}", image_name);
        original_path = format!("{}/{}", cache_dir, image_name);

        let file_result: Result<fs::File, Error> = web::block({
            let original_path: String = original_path.clone();
            move || File::create(&original_path)
        })
        .await
        .expect("Failed to execute blocking operation");

        let mut f: fs::File = match file_result {
            Ok(file) => file,
            Err(e) => {
                error!("Error creating file {}: {:?}", original_path, e);
                status_map.insert(request_id.clone(), "Error".to_string());
                return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
            }
        };
        info!("Saving file to: {}", original_path);
        while let Some(chunk) = field.next().await {
            let data: web::Bytes = match chunk {
                Ok(data) => data,
                Err(e) => {
                    error!("Error reading chunk: {:?}", e);
                    status_map.insert(request_id.clone(), "Error".to_string());
                    return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
                }
            };
            if let Err(e) = web::block({
                let mut f = f.try_clone().expect("Failed to clone file handle");
                move || {
                    std::io::Write::write_all(&mut f, &data)
                        .map(|_| f)
                }
            })
            .await
            {
                error!("Error writing data to file: {:?}", e);
                status_map.insert(request_id.clone(), "Error".to_string());
                return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
            }
        }
    }

    use tokio::process::Command;
    use tokio::io::{self, AsyncBufReadExt};
    use std::process::Stdio;

    // Determine the file extension from the original image name
    let extension = Path::new(&image_name)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("jpg");

    // Update the upscaled path to use the request_id and the determined extension
    upscaled_path = format!("{}/{}.{}", cache_dir, request_id, extension);

    info!("Starting upscaling process for: {}", original_path);
    #[cfg(windows)]
    let command_path: &str = "./realesrgan-ncnn-windows/realesrgan-ncnn-vulkan.exe";
    #[cfg(not(windows))]
    let command_path: &str = "./realesrgan-ncnn-ubuntu/realesrgan-ncnn-vulkan";
    let args: [&str; 8] = [
        "-i",
        &original_path,
        "-o",
        &upscaled_path,
        "-g",
        "0",
        "-s",
        "2",
    ];
    info!("Command path: {}", command_path);
    info!("Command args: {:#?}", args);
    let mut command: Command = {
        let mut cmd: Command = Command::new(command_path);
        cmd.args(&args);
        cmd.stdout(Stdio::piped());
        cmd
    };
    trace!("Running command: {:?}", command);

    let mut child: tokio::process::Child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            error!("Failed to spawn command: {:?}", e);
            if let Some(13) = e.raw_os_error() {
                error!("Permission denied: Please check the permissions of the command or the file paths.");
            }
            let mut data_lock: MutexGuard<'_, AppState> = data.lock().await;
            let mut status_map: MutexGuard<'_, HashMap<String, String>> =
                data_lock.request_status.lock().await;
            status_map.insert(request_id.clone(), "Error".to_string());
            return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
        }
    };

    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            error!("Failed to open stdout");
            status_map.insert(request_id.clone(), "Error".to_string());
            return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
        }
    };

    let reader = io::BufReader::new(stdout);
    let mut lines = reader.lines();

    while let Some(line) = match lines.next_line().await {
        Ok(line) => line,
        Err(e) => {
            error!("Failed to read line: {:?}", e);
            status_map.insert(request_id.clone(), "Error".to_string());
            return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
        }
    } {
        println!("Upscaling process output: {}", line);
    }

    // Check if the upscaled file exists before waiting for the command to finish
    if Path::new(&upscaled_path).exists() {
        info!("Upscaling successful, file saved to: {}", upscaled_path);
        status_map.insert(request_id.clone(), "Completed".to_string());
        return HttpResponse::Ok().json(json!({"status": "Completed", "data": {"request_id": request_id, "upscaled_path": upscaled_path}}));
    }

    let output = match child.wait_with_output().await {
        Ok(output) => output,
        Err(e) => {
            error!("Error waiting for command output: {:?}", e);
            status_map.insert(request_id.clone(), "Error".to_string());
            return HttpResponse::InternalServerError().json(json!({"status": "Error", "data": {"request_id": request_id}}));
        }
    };

    let mut data_lock: MutexGuard<'_, AppState> = data.lock().await;
    let mut status_map: MutexGuard<'_, HashMap<String, String>> =
        data_lock.request_status.lock().await;

    let status = if output.status.success() {
        if Path::new(&upscaled_path).exists() {
            info!("Upscaling successful, file saved to: {}", upscaled_path);
            status_map.insert(request_id.clone(), "Completed".to_string());
            "Completed"
        } else {
            error!("Upscaling failed, upscaled file not found");
            status_map.insert(request_id.clone(), "Failed".to_string());
            "Failed"
        }
    } else {
        error!("Upscaling process exited with error: {:?}", output.status);
        status_map.insert(request_id.clone(), "Error".to_string());
        "Error"
    };

    HttpResponse::Ok().json(json!({"status": status, "data": {"request_id": request_id, "upscaled_path": upscaled_path}}))
}

async fn get_status(data: Data<Mutex<AppState>>, request_id: web::Path<String>) -> impl Responder {
    let request_id: String = request_id.into_inner();
    let status_map_lock: MutexGuard<'_, AppState> = data.lock().await;
    let status_map: MutexGuard<'_, HashMap<String, String>> =
        status_map_lock.request_status.lock().await;
    if let Some(status) = status_map.get(&request_id) {
        HttpResponse::Ok().json(json!({"status": status}))
    } else {
        HttpResponse::NotFound().body("Request ID not found")
    }
}

async fn ping() -> impl Responder {
    info!("Received ping request");
    HttpResponse::Ok().json(json!({"status": "healthy", "data": "pong"}))
}

#[actix_web::main]
async fn main() -> ioResult<()> {
    info!("Starting server");
    init_tracing();

    #[cfg(windows)]
    let cache_dir: &str = "C:/Users/floris/Documents/Github/esrgan/cache";
    
    #[cfg(unix)]
    let cache_dir: &str = "/home/floris-xlx/repos/esrgan/cache";
    fs::create_dir_all(cache_dir)?;
    info!("Cache directory created at: {}", cache_dir);

    let app_state: Data<Mutex<AppState>> = Data::new(Mutex::new(AppState {
        cache_dir: cache_dir.to_string(),
        request_status: Mutex::new(HashMap::new()),
    }));

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .route("/upscale", web::post().to(upscale_image_post))
            .route("/status/{request_id}", web::get().to(get_status))
            .route("/ping", web::get().to(ping))
    })
    .bind("127.0.0.1:3443")?
    .run()
    .await
}

/// ## Initialize Tracing
///
/// This function sets up the tracing subscriber for logging and monitoring.
///
/// ### Example
///
/// ```
/// init_tracing();
/// ```
fn init_tracing() {
    let filter: EnvFilter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(filter).init()
}

use askama::Template;

#[derive(Template)]
#[template(path = "template.html")]
pub struct ExportHtml {
    pub image: String,
}

import streamlit as st


def edit_test_form(test_obj, test_type):
    st.markdown("### Edit Test Fields")
    with st.form("edit_test_fields"):
        type_fields = {}
        type_fields["max_diffs"] = st.number_input(
            "Max Diffs", value=test_obj.max_diffs, min_value=0, step=1
        )
        type_fields["unidecode"] = st.checkbox("Unidecode", value=test_obj.unidecode)
        type_fields["alphanum"] = st.checkbox("Alphanum", value=test_obj.alphanum)
        type_fields["ignore_str"] = st.text_input(
            "Ignore strings (seperarated by spaces)",
            value=" ".join(test_obj.ignore_str),
        )
        type_fields["ignore_space"] = st.checkbox(
            "Ignore space", value=test_obj.ignore_space
        )

        type_fields["ignore_str"] = (
            type_fields["ignore_str"].split(" ") if type_fields["ignore_str"] else []
        )

        if test_type == "present" or test_type == "absent":
            type_fields["text"] = st.text_area(
                "Text", value=test_obj.text, height="content"
            )
            layout_cat_options = [
                "text",
                "footer",
                "header",
                "footnote",
                "image",
                "image_caption",
            ]

            type_fields["layout_cat"] = st.selectbox(
                "Layout Category",
                layout_cat_options,
                index=layout_cat_options.index(test_obj.layout_cat),
            )
            type_fields["case_sensitive"] = st.checkbox(
                "Case Sensitive", value=test_obj.case_sensitive
            )
            type_fields["first_n"] = st.number_input(
                "First N",
                value=test_obj.first_n if test_obj.first_n else 0,
                min_value=0,
                step=100,
            )
            type_fields["last_n"] = st.number_input(
                "Last N",
                value=test_obj.last_n if test_obj.last_n else 0,
                min_value=0,
                step=100,
            )
            if type_fields["first_n"] == 0:
                type_fields["first_n"] = None
            if type_fields["last_n"] == 0:
                type_fields["last_n"] = None
        elif test_type == "order":
            type_fields["before"] = st.text_area(
                "Before", value=test_obj.before, height="content"
            )
            type_fields["after"] = st.text_area(
                "After", value=test_obj.after, height="content"
            )
        elif test_type == "table":
            type_fields["cell"] = st.text_input("Cell", value=test_obj.cell)
            type_fields["up"] = st.text_input(
                "Up", value=test_obj.up if test_obj.up else ""
            )
            type_fields["down"] = st.text_input(
                "Down", value=test_obj.down if test_obj.down else ""
            )
            type_fields["left"] = st.text_input(
                "Left", value=test_obj.left if test_obj.left else ""
            )
            type_fields["right"] = st.text_input(
                "Right", value=test_obj.right if test_obj.right else ""
            )
            type_fields["top_heading"] = st.text_input(
                "Top Heading",
                value=test_obj.top_heading if test_obj.top_heading else "",
            )
            type_fields["left_heading"] = st.text_input(
                "Left Heading",
                value=test_obj.left_heading if test_obj.left_heading else "",
            )
        if st.form_submit_button("Save Changes"):
            for field, value in type_fields.items():
                setattr(test_obj, field, value)

            return test_obj

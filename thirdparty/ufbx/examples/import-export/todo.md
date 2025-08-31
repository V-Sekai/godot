1. `-Wmissing-braces` fix errors in compile
2. FIXME: material->element.element_id = scene_imp->num_materials + 2000; // Offset to avoid ID conflicts
   Offsets seem small... for example 2000 materials will overflow.
